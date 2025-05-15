import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, label_fn=None):
        """
        Args:
            img_dir (str): 图片根目录路径
            transform (callable, optional): 图像预处理变换
            label_fn (callable, optional): 自定义标签提取函数
                                           默认使用父文件夹名作为标签
        """
        self.img_dir = img_dir
        self.transform = transform
        self.label_fn = label_fn or self._default_label_fn

        # 收集所有图片路径及其对应标签
        self.samples = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    label = self.label_fn(img_path)
                    self.samples.append((img_path, label))

    def _default_label_fn(self, path):
        """默认标签提取函数：使用父文件夹名作为标签"""
        return os.path.basename(os.path.dirname(path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图片并确保为RGB格式
        image = Image.open(img_path).convert('L')
        # image = Image.fromarray(image.numpy(), mode="L")

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, label


# 使用示例
if __name__ == "__main__":
    import torchvision.transforms as T

    # 初始化数据集（假设图片按类别存放在子文件夹中）
    dataset = CustomImageDataset(
        img_dir='path/to/your/images',
        transform=T.Compose([
            T.Resize((224, 224)),
            # 如需转为Tensor，可添加：T.ToTensor()
        ])
    )

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # 验证使用
    print(f"数据集总样本数: {len(dataset)}")
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"\n批次 {batch_idx}:")
        print(f"图片类型: {type(images[0])}")  # 应显示PIL.Image.Image
        print(f"首张图片尺寸: {images[0].size}")
        print(f"示例标签: {labels[0]}")
        if batch_idx == 0: break