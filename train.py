import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import Encoder
import Decoder
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
        img_name = os.path.join(self.input_dir, "GT" + self.images[idx] + ".png")
        gt_name = os.path.join(self.gt_dir, "GT" + self.images[idx] + ".png") 
        
        image = Image.open(img_name).convert("RGB")
        gt_image = Image.open(gt_name).convert("L") 

        return image, gt_image


train_dataset = AlphaMattingDataset(input_dir='Data/Train/InputImages', gt_dir='Data/Train/GroundTruthAlphas', trimaps = 'Data/Train/Trimaps')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

encoder = Encoder.Encoder()
decoder = Decoder.Decoder()
discriminator = Discriminator.PatchGANDiscriminator(input_channels=3) 