# AlphaGAN

Reimplementation of the AlphaGAN architecture

Original Paper: https://arxiv.org/pdf/1807.10088.pdf 
Authors: Sebastian Lutz, Konstantinos Amplianitis, Aljoša Smolic

# Download Data

Create the following directorires

```sh
mkdir -p Data/Train/InputImages
mkdir -p Data/Train/Trimaps
mkdir -p Data/Train/GroundTruthAlphas
mkdir -p Data/Test/InputImages
mkdir -p Data/Test/Trimaps
```

Use the following commands to download and unzip the data

```sh
# Train Data
curl -o InputImages.zip http://alphamatting.com/datasets/zip/input_training_lowres.zip ; unzip InputImages.zip -d ./Data/Train/InputImages
curl -o Trimaps.zip http://alphamatting.com/datasets/zip/trimap_training_lowres.zip ; unzip Trimaps.zip -d ./Data/Train/Trimaps
curl -o GroundTruthAlphas.zip http://alphamatting.com/datasets/zip/gt_training_lowres.zip ; unzip GroundTruthAlphas.zip -d ./Data/Train/GroundTruthAlphas
# Test Data
curl -o TestInputImages.zip http://alphamatting.com/datasets/zip/input_lowres.zip ; unzip TestInputImages.zip -d ./Data/Test/InputImages
curl -o TestTrimaps.zip http://alphamatting.com/datasets/zip/trimap_lowres.zip ; unzip TestTrimaps.zip -d ./Data/Test/Trimaps
```


# Create Virtual Environment

```
# Create virtual env
python3.11 -m venv venv
source venv/bin/activate
# Download the requirements for project
pip -r requirements.txt
```
