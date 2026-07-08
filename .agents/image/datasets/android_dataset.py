import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_transform(image_size, belong='train'):
    if belong == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(image_size[0] / 0.875), int(image_size[1] / 0.875))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(image_size[0] / 0.875), int(image_size[1] / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class AndroidDataset(Dataset):
    """
    # Description:
        针对 imb_android 数据集的加载器
        结构：imb_android -> {train, val, test} -> {mal, leg} -> images
        
    """

    def __init__(self,DATAPATH,belong='train', image_size=(448, 448)):
        """
            DATAPATH: 数据集父路径
            belong: 是train还是val还是test
            image_size: 图片大小

        """
        assert belong in ['train', 'val', 'test']
        self.belong = belong
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        
        self.class_to_idx = {'benign': 0, 'mal': 1}
        self.class_aliases = {
            0: ['benign', 'leg', 'normal', '0'],
            1: ['mal', 'malware', '1'],
        }
        self.num_classes = 2
        
        self.image_samples = [] # 存储格式: (image_full_path, label_idx)

        # 拼接当前阶段的文件夹路径，例如: ./imb_android/train
        belong_dir = os.path.join(DATAPATH, self.belong)
        
        seen_paths = set()
        for label, aliases in self.class_aliases.items():
            for class_name in aliases:
                class_dir = os.path.join(belong_dir, class_name)
                if not os.path.exists(class_dir):
                    continue

                for root, _, fnames in sorted(os.walk(class_dir)):
                    for fname in sorted(fnames):
                        if self._is_image_file(fname):
                            path = os.path.abspath(os.path.join(root, fname))
                            if path in seen_paths:
                                continue
                            seen_paths.add(path)
                            self.image_samples.append((path, int(label)))

        # 获取数据增强转换
        self.transform = get_transform(self.image_size, self.belong)

    def _is_image_file(self, filename):
        """检查是否为常见图片格式"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

    def __getitem__(self, item):
        # 获取路径和标签
        img_path, label = self.image_samples[item]

        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_samples)

    def get_image_path(self, item):
        return self.image_samples[item][0]

    def get_sample_info(self, item):
        image_path, label = self.image_samples[item]
        path = Path(image_path)
        sample_id = path.stem
        return {
            'sample_id': sample_id,
            'apk_name': f'{sample_id}.apk',
            'split': self.belong,
            'label': int(label),
            'image_path': str(path),
        }

def cal_prior(dataset):
    """
    计算数据集的对数先验概率
    
    Args:
        dataset: PyTorch Dataset对象，每个元素应为 (data, label) 格式
        
    Returns:
        log_prior: 对数先验概率张量，形状为 (num_classes,)
        class_weights: 类别权重（可选）
    """
    # 收集所有标签
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]  # 假设每个样本是 (data, label)
        labels.append(label)
    
    # 转换为张量
    labels = torch.tensor(labels)
    
    # 计算类别分布
    num_classes = len(torch.unique(labels))
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    
    # 计算先验概率 P(class) = count(class) / total_samples
    priors = class_counts.float() / total_samples
    
    # 计算对数先验概率
    log_prior = torch.log(priors + 1e-10)  # 添加小常数避免log(0)
    
    return log_prior

if __name__ == '__main__':
    DATAPATH = '/home/linux/7T/lzw/datasets/android_zoo/android_dex_images'
    train_dataset= AndroidDataset(DATAPATH, belong='train', image_size=512)
    prior = cal_prior(train_dataset)
    # 打印计算出来的先验
    print(prior)
    # tensor([-0.3945, -1.1210])
