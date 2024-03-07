import os
from PIL import Image
from pathlib import Path
import torch as t
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import Generator
import Discriminator


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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

class TestDataset(Dataset):
    def __init__(self, input_dir, trimap_dir, transform=None):
        self.input_dir = Path(input_dir)
        self.trimap_dir = Path(trimap_dir)
        self.transform = transform
        self.images = list(self.input_dir.glob('*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = img_path.stem
        img = Image.open(img_path).convert("RGB")
        trimap = Image.open(self.trimap_dir / (img_name + '.png')).convert("L")

        if self.transform:
            img = self.transform(img)
            trimap = self.transform(trimap)

        return img, trimap
    
transformer = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(320)])

# Training Data
train_dataset = AlphaMattingDataset(input_dir='Data/Train/InputImages', gt_dir='Data/Train/GroundTruthAlphas',
                                    trimaps='Data/Train/Trimaps')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Test Dataset
test_dataset = TestDataset(input_dir="Data/Test/Input Image", trimap_dir="Data/Test/TriMaps", transform=transformer)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


device = t.device("cuda" if t.cuda.is_available() else "cpu")

epochs = 1
lr_g = 0.001
lr_d = 0.001


if t.cuda.is_available():
    device = t.device("cuda")
# elif t.backends.mps.is_built():
#     device = t.device("mps")
else:
    device = t.device("cpu")


generator = Generator.Generator().to(device)
discriminator = Discriminator.PatchGANDiscriminator(patch_size=4).to(device)

optim_g = t.optim.Adam(generator.parameters(), lr=0.001)
optim_d = t.optim.Adam(discriminator.parameters(), lr=0.001)

l1_loss = t.nn.SmoothL1Loss().to(device)
mse_loss = t.nn.MSELoss().to(device)

for epoch in range(epochs):
    for image, tri_image, gt_image in train_loader:
        image = image.to(device)
        tri_image = tri_image.to(device)
        gt_image = gt_image.to(device)
        input_img = t.cat((image, tri_image), dim=1).to(device)

        # Train discriminator
        optim_d.zero_grad()

        real_alpha_pred = discriminator(t.cat((image, gt_image), dim=1))
        d_real_loss = mse_loss(real_alpha_pred, t.ones_like(real_alpha_pred))

        fake_alpha = generator(input_img)
        fake_alpha_pred = discriminator(t.cat((image, fake_alpha.detach()), dim=1))
        d_fake_loss = mse_loss(fake_alpha_pred, t.zeros_like(fake_alpha_pred))

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optim_d.step()

        # Train generator
        optim_g.zero_grad()

        fake_alpha = generator(input_img)
        g_l1_loss = l1_loss(fake_alpha, gt_image)

        fake_alpha_pred = discriminator(t.cat((image, fake_alpha), dim=1))
        g_gan_loss = mse_loss(fake_alpha_pred, t.ones_like(fake_alpha_pred))

        g_loss = g_l1_loss + g_gan_loss
        g_loss.backward()
        optim_g.step()

    print(f"Epoch {epoch + 1}")
