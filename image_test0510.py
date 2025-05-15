import os
import torchvision
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from utils.networkHelper import *

from UNet import Unet
from utils.trainNetworkHelper import SimpleDiffusionTrainer
from simpleDiffusion import DiffusionModel
from datasets import CustomImageDataset


# 数据集加载
# data_root_path = r"E:\charmcode\GAN\diffusion-torch\CNV"
# if not os.path.exists(data_root_path):
#     os.makedirs(data_root_path)


custom_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整尺寸
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.ToTensor()
])

# imagenet_data = torchvision.datasets.MNIST(data_root_path, train=True, download=True, transform=transforms.ToTensor())
dataset = CustomImageDataset(
    img_dir=r"E:\charmcode\GAN\diffusion-torch\CNV",
    transform=custom_transform,
)


image_size = 128
channels = 1
batch_size = 4


data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
dim_mults = (1, 2, 4,)

denoise_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)

timesteps = 1000
schedule_name = "linear_beta_schedule"
DDPM = DiffusionModel(schedule_name=schedule_name,
                      timesteps=timesteps,
                      beta_start=0.0001,
                      beta_end=0.02,
                      denoise_model=denoise_model).to(device)

optimizer = Adam(DDPM.parameters(), lr=1e-3)

epoches = 100

Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)

# 训练参数设置
root_path = "./saved_train_models"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(image_size, channels, dim_mults, timesteps, schedule_name)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)


# 训练好的模型加载，如果模型是已经训练好的，则可以将下面两行代码取消注释
best_model_path = saved_path + '/' + 'BestModel.pth'
DDPM.load_state_dict(torch.load(best_model_path))

# 如果模型已经训练好则注释下面这行代码，反之则注释上面两行代码
# DDPM = Trainer(DDPM, model_save_path=saved_path)

# 采样:sample 64 images
bs = 16
samples = DDPM(mode="generate", image_size=image_size, batch_size=bs, channels=channels)

# 随机挑一张显示
random_index = 1
for i in range(bs):

    generate_image = samples[-1][i].reshape(channels, image_size, image_size)
    figtest = reverse_transform(torch.from_numpy(generate_image))
    figtest.save(f"./outimage/{i}.jpg")
# figtest.show()


