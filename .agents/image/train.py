import os
import time
import argparse
import logging
import warnings
from datetime import datetime
import wandb
os.environ["WANDB_MODE"] = "offline"
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import IDDCNF
from datasets import AndroidDataset
from utils import get_feature_contrastive_loss, FeatureMetric,get_flow_contrastive_loss,validate, get_orthogonal_mask_loss,send_email
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score,roc_auc_score,average_precision_score, recall_score,confusion_matrix,classification_report

def main(args):
    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    now_time = datetime.now().strftime('%m-%d_%H-%M')
    # args.save_dir = os.path.join(args.save_dir,now_time)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(
        filename=os.path.join(args.save_dir, args.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")
    logging.info('本次训练的信息：'+f'{args.info}')
    logging.info('本次训练使用GPU：'+f'{args.gpu}')

    wandb.init(project=args.wandb_name, config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "image_size":args.image_size,
        "latent_weight":args.latent_weight,
        "feature_contrastive_weight":args.feature_contrastive_weight,
        "flow_contrastive_weight":args.flow_contrastive_weight,
        "feature_margin":args.feature_margin,
        "flow_margin":args.flow_margin
    })


    DATAPATH = args.data_path
    train_dataset = AndroidDataset(DATAPATH,image_size=args.image_size,belong='train')
    print(f"class mapping：{train_dataset.class_to_idx}")
    validate_dataset = AndroidDataset(DATAPATH,image_size=args.image_size,belong='val')
    # 先验概率，由于贝叶斯分类的特殊性
    prior= None
    if args.use_prior:
        print('use prior!')
        prior = torch.tensor([-0.3945, -1.1210]).to(device)
    else:
        prior= None
    args.prior = prior

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size * 4, shuffle=False,num_workers=args.workers, pin_memory=True)
    num_classes = args.num_classes

    logs = {}
    model = IDDCNF(num_classes=num_classes, num_masks=args.num_masks, num_attn_layers=args.num_attn_layers,latent_input_dim=args.latent_input_dim)

    # 如果是恢复训练的化，这个start_epoch就是恢复的epoch
    start_epoch = 0
    # if args.ckpt:
    #     checkpoint = torch.load(args.ckpt)
    #     logs = checkpoint['logs']
    #     start_epoch = int(logs['epoch'])
    #     state_dict = checkpoint['state_dict']
    #     model.load_state_dict(state_dict)
    #     logging.info('Network loaded from {}'.format(args.ckpt))

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # # 初始化用于计算特征图像对比损失的距离的工具类,
    feature_metric = FeatureMetric(dim=model.num_features).to(device)  

    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(feature_metric.parameters()),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-5
    )

    # 学习率调度，采用cosine    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs//2, 
        eta_min=1e-6
    )

    print('Training dataset size: {}, Validation dataset size: {}'.
                 format(len(train_dataset), len(validate_dataset)))
    # logging.info('')

    # early stop逻辑变量
    temp_patience = 0
    logs['best_acc'] = 0
    logs['best_pr_auc'] = 0
    pbar = tqdm(range(start_epoch, args.epochs), 
            desc='IDDCNF training!',
            initial=start_epoch,
            total=args.epochs)

    for epoch in pbar:
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']
        # 训练一次
        train_one_epoch(logs=logs,feature_metric=feature_metric,data_loader=train_loader,model=model,optimizer=optimizer, device=device, args=args)

        # 每轮都验证一次
        validate(logs,validate_loader,model,device,args)

        # 记录当前学习率
        wandb.log({
                "epoch": logs["epoch"],
                "lr": logs["lr"],
            },step=logs['epoch'])

        # 更新模型的逻辑。只要pr_auc有提升，就保存模型
        if logs['val_pr_auc'] > logs['best_pr_auc']:
            logs['best_pr_auc'] = logs['val_pr_auc']
            logs['best_acc'] = logs['val_acc']
            best_final_ckpt_path = os.path.join(args.save_dir, "best_final_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'logs': logs,
            }, best_final_ckpt_path)
            print("\n"+f"epoch:{epoch+1},Best final model updated! New best pr auc = {logs['best_pr_auc']:.2f}%")
            # 重置
            temp_patience = 0
        else:
            # 减少耐心
            temp_patience += 1
            print(f"\nEpoch {epoch+1}: No improvement in pr_auc. Patience: {temp_patience}/{args.patience}")
            
            if temp_patience>args.patience:
                print('\n\nearly stop!all train finished!\n\n')
                print(f'best val acc:{logs["best_acc"]:.2f},val pr auc:{logs["best_pr_auc"]:.2f}')
                break

        # CosineAnnealingLR 会在这里根据当前 epoch 自动计算余弦值
        scheduler.step()
    
    pbar.close()
    # 训练结束，保存最终模型
    last_ckpt_path = os.path.join(args.save_dir, "last_model.pth")
    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'logs': logs,
        }, last_ckpt_path)
    print(f"Last model saved at epoch {epoch + 1}")
    
    return logs
    # 测试训练集的指标，查看是否已经训练完毕，还是缺乏训练次数
    # validate(logs=logs,data_loader=train_loader,model=model,prior=prior,device=device,alpha=args.alpha)
    # print('\nuse train dataset to validate,the results:\n'+f'acc={logs["val_acc"]:.2f},precise={logs["val_precise"]:.2f},recall={logs["val_recall"]:.2f},f1={logs["val_f1"]:.2f},pr_auc={logs["val_pr_auc"]:.2f},roc_auc={logs["val_roc_auc"]:.2f},low_fpr_auc={logs["val_low_fpr_auc"]:.2f}')
    


def train_one_epoch(logs, data_loader,feature_metric, model, optimizer, device, args):

    # 加载各个部分的权重
    latent_weight=args.latent_weight
    flow_contrastive_weight=args.flow_contrastive_weight
    feature_contrastive_weight=args.feature_contrastive_weight
    ortho_weight = args.ortho_weight
    
    if logs['epoch'] == 1:
        print(f"当前权重设置：latent_weight={latent_weight}, flow_contrastive_weight={flow_contrastive_weight}, feature_contrastive_weight={feature_contrastive_weight}, ortho_weight={ortho_weight}")

    model.train()
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        cls_features,attention_maps, z_gc, log_det = model(images)
                
        # 正交掩码正则化损失
        loss_ortho = get_orthogonal_mask_loss(attention_maps)
        # 图像的对比损失，核心目标:在特征层面，同类特征拉近+异类特征拉远, 条件映射到混合高斯模型的损失,核心目的，在隐空间中，要能分得开类1和类2
        loss_feature_contrast = get_feature_contrastive_loss(cls_features, labels, feature_metric,feature_margin=args.feature_margin)
        # 流对比损失（最大化联合概率），核心目标：最大化各自类别P(y|x)的概率
        loss_latent, loss_nll, loss_contrast = get_flow_contrastive_loss(z_gc, log_det, model.mu_gc, model.log_var_gc,labels,class_prior=args.prior,flow_margin=args.flow_margin)
        # 加权总损失
        loss = ortho_weight * loss_ortho  \
            + latent_weight * loss_latent \
            + flow_contrastive_weight * (loss_nll+loss_contrast) \
            + feature_contrastive_weight * loss_feature_contrast
        
        wandb.log({
            'latent_loss':loss_latent,
            'loss_flow_contrast':(loss_nll+loss_contrast),
            'loss_feature_contrast':loss_feature_contrast,
            'loss_ortho':loss_ortho,
            'loss':loss,
        },step=logs['epoch'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(feature_metric.parameters()),
            max_norm=5.0
        )
        optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training configuration for IDDCNF")

    ##################################################
    # Training Config
    ##################################################
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--latent_input_dim', type=int, default=32)

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--patience', type=int, default=10)

    ##################################################
    # Finetune Params
    ##################################################
    parser.add_argument('--latent_weight', type=float, default=0.5)
    # ⭐ flow对比损失权重（你要改的）
    parser.add_argument('--flow_contrastive_weight', type=float, default=0.0001)
    parser.add_argument('--feature_contrastive_weight', type=float, default=0.05)
    parser.add_argument('--ortho_weight', type=float, default=0.1)
    parser.add_argument('--feature_margin', type=float, default=1.0,help='margin for feature contrastive loss')
    parser.add_argument('--flow_margin', type=float, default=1.0,help='margin for flow contrastive loss')

    ##################################################
    # Model Config
    ##################################################
    parser.add_argument('--image_size', type=int, nargs=2, default=(448, 448))
    parser.add_argument('--num_masks', type=int, default=32)

    ##################################################
    # Dataset / Path Config
    ##################################################
    parser.add_argument('--data_path', type=str,
                        default='/home/user/lizuwang/datasets/imb_android')
    parser.add_argument('--save_dir', type=str, default='runs/train')
    parser.add_argument('--log_name', type=str, default='train.log')

    ##################################################
    # Eval Config
    ##################################################
    parser.add_argument('--eval_ckpt', type=str,
                        default='/home/user/lizuwang/BIDO/runs/train/03-23_19-34/best_final_model.pth')
    parser.add_argument('--eval_savepath', type=str, default='./classify')

    ##################################################
    # Info
    ##################################################
    parser.add_argument('--info', type=str,
                        default="",help='助记信息')
    parser.add_argument('--wandb_name', type=str,
                        default="IDDCNF",help='wandb项目名称')                       

    args = parser.parse_args()
    logs = main(args)
    send_email(f"Training completed successfully {logs}")

