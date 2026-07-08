import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score,roc_auc_score,average_precision_score, recall_score,confusion_matrix,classification_report
import os
import time
import argparse
import logging
import warnings
from datetime import datetime
import wandb
from tqdm import tqdm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

def validate(logs,data_loader,model,device,args):
    prior = args.prior

    all_preds = []
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            _, _, z_gc, log_det = model(images)
            # --- 2. 获取贝叶斯分支的概率 (Generative) ---
            num_classes = model.num_classes
            latent_dim = z_gc.shape[1]
            # 向量化计算所有类别
            z_expand = z_gc.unsqueeze(1)  # [B, 1, D]
            mu_expand = model.mu_gc.unsqueeze(0)  # [1, K, D]
            
            # [修改处] 引入推断时的异方差计算
            log_var_clamped = torch.clamp(model.log_var_gc, min=-10.0, max=10.0)
            var_expand = torch.exp(log_var_clamped).unsqueeze(0) # [1, K, D]
            dist_sq_mahalanobis = torch.sum(((z_expand - mu_expand) ** 2) / var_expand, dim=-1) # [B, K]
            log_det_var = torch.sum(log_var_clamped, dim=-1).unsqueeze(0) # [1, K]
            
            log_p_z_y = -0.5 * (dist_sq_mahalanobis + log_det_var)  # [B, K]
            # log p(x|y) = log p(z|y) + log|det|
            log_p_x_y = log_p_z_y + log_det.unsqueeze(1)  # [B, K]
            # 添加先验 log p(y)
            if prior is not None:
                # prior 应该是 log p(y) 的形式
                log_prior = prior  # [K]
            else:
                # 均匀先验：log(1/K)
                log_prior = -torch.log(torch.tensor(num_classes, dtype=torch.float, device=z_gc.device))
            # log p(x, y) = log p(x|y) + log p(y) = log p(y|x) + log p(x),log p(x, y),表明log p(y|x)越大，达到分类效果
            # 如果对一个样本的log p(x, y)（即log p(y|x)）很小，说明这个图片处于分布外
            joint_log_likelihood = log_p_x_y + log_prior.unsqueeze(0)  # [B, K]
            # 后验概率 p(y|x) = softmax(log p(x, y))
            bayes_probs = F.softmax(joint_log_likelihood/latent_dim, dim=1)
            # --- 3. 概率加权融合 (索引 1 为要检测的正类) ---
            # 融合两者对类别 1 的预测概率
            combine_probs = bayes_probs[:, 1]
            # 收集真实标签和融合后的概率
            all_labels.append(labels.cpu())
            all_probs.append(combine_probs.cpu())
    # ================= 数据整理 =================
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    # print(f"all labels:{all_labels}")
    print(f"all probs:{all_probs}")
    mal_index=0
    # for i,p in enumerate(all_probs):
    #     if all_labels[i]==1:
    #         mal_index+=1
    #     if p<0.5 and all_labels[i]==1:
    #         # print(f"wrong predict index {i}:{p},path{test_dataset.get_image_path(i)}")
    #         print("\""+f"{test_dataset.get_image_path(i)}"+"\",")
    # ================= 阈值 =================
    best_thresh = 0.5
    final_preds = (all_probs >= best_thresh).astype(int)
    # ================= 混淆矩阵 =================
    confusion_mtx = confusion_matrix(all_labels, final_preds)
    print("Confusion Matrix:")
    print(confusion_mtx)
    # 防止出现只有一个类别导致报错
    if confusion_mtx.shape == (2, 2):
        TN, FP, FN, TP = confusion_mtx.ravel()
    else:
        # 极端情况（测试集只有一类）
        TN = FP = FN = TP = 0
        if np.unique(all_labels)[0] == 0:
            TN = confusion_mtx[0, 0]
        else:
            TP = confusion_mtx[0, 0]
    # ================= 基础指标 =================
    acc = accuracy_score(all_labels, final_preds) * 100
    # ROC AUC & PR AUC（用概率）
    try:
        roc_auc = roc_auc_score(all_labels, all_probs) * 100
        pr_auc = average_precision_score(all_labels, all_probs) * 100
    except:
        roc_auc, pr_auc = 0.0, 0.0
    # ================= ACSA 和 GM =================
    # Recall for each class
    eps = 1e-12
    recall_pos = TP / (TP + FN + eps)   # 正类召回
    recall_neg = TN / (TN + FP + eps)   # 负类召回
    # ACSA (Average Class Specific Accuracy)
    acsa = (recall_pos + recall_neg) / 2 * 100
    # GM (Geometric Mean)
    gm = np.sqrt(recall_pos * recall_neg) * 100
    # ================= 分类报告 =================
    class_report = classification_report(all_labels, final_preds, digits=4)
    print("\nClassification Report:")
    print(class_report)
    # ================= 最终打印 =================
    print('\n[Final Metrics]')
    print('ACC      : {:.4f}'.format(acc))
    print('ROC AUC  : {:.4f}'.format(roc_auc))
    print('PR AUC   : {:.4f}'.format(pr_auc))
    print('ACSA     : {:.4f}'.format(acsa))
    print('GM       : {:.4f}'.format(gm))

    logs['val_pr_auc'] = pr_auc
    logs['val_acc'] = acc
    logs['val_roc_auc'] = roc_auc
    logs['val_acsa'] = acsa
    logs['val_gm'] = gm


def send_email(text):
    # 发件人邮箱
    sender = "2570693148@qq.com"

    # QQ邮箱SMTP授权码
    password = "ddkyfpderekueadc"

    # 收件人邮箱
    receiver = "2570693148@qq.com"

    # 邮件内容
    message = MIMEMultipart()

    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = Header("Python任务执行完成通知", "utf-8")

    message.attach(MIMEText(text, "plain", "utf-8"))

    try:
        # QQ邮箱SMTP服务器
        smtp = smtplib.SMTP_SSL("smtp.qq.com", 465)

        smtp.login(sender, password)

        smtp.sendmail(
            sender,
            receiver,
            message.as_string()
        )

        smtp.quit()

        print("邮件发送成功")

    except Exception as e:
        print("邮件发送失败：", e)