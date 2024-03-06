import os
import numpy as np
from PIL import Image
import torch as t
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import Generator
import Discriminator


class AlphaMattingDataset(Dataset):
    def __init__(self, input_dir, gt_dir, trimaps):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.trimaps = trimaps
        self.images = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.input_dir, self.images[idx])
        tri_name = os.path.join(self.trimaps, "Trimap1/" + self.images[idx])
        gt_name = os.path.join(self.gt_dir, self.images[idx])

        image = Image.open(img_name).convert("RGB")
        tri_image = Image.open(tri_name).convert("L")
        gt_image = Image.open(gt_name).convert("L")

        image = transformer(image)
        tri_image = transformer(tri_image)
        gt_image = transformer(gt_image)

        return image, tri_image, gt_image


transformer = transforms.Compose([transforms.ToTensor()])

train_dataset = AlphaMattingDataset(input_dir='Data/Train/InputImages', gt_dir='Data/Train/GroundTruthAlphas',
                                    trimaps='Data/Train/Trimaps')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

generator = Generator.Generator()
discriminator = Discriminator.PatchGANDiscriminator(input_channels=3)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

for image, tri_image, gt_image in train_dataset:
    input_img = t.tensor(np.concatenate((image.numpy(), tri_image.numpy()), axis=0)).to(device)
    gt_img = gt_image.to(device)
