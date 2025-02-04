# SplitFed: When Federated Learning Meets Split Learning

Releasing the source code version1 of our work "SplitFed: When Federated Learning Meets Split Learning."

We have three versions of our programs:

Version1: without using socket and no DP+PixelDP

Version2: with using socket but no DP+PixelDP

Version3: without using socket but with DP+PixelDP (required more packages)

Other versions will be released soon.

Please cite the main paper if useful: https://arxiv.org/pdf/2004.12088.pdf


## Description

This repository contains the implementation of Centralized Learning (baseline), Federated Learning, Split Learning, SplitFedV1 Learning and SplitFedV2 Learning.

All programs are written in python 3.7.2 using the PyTorch library (PyTorch 1.2.0).

Dataset: HAM10000

Model: ResNet18

# SplitFed: When Federated Learning Meets Split Learning

Releasing the source code version1 of our work "SplitFed: When Federated Learning Meets Split Learning."

We have three versions of our programs:

- **Version1**: without using socket and no DP+PixelDP
- **Version2**: with using socket but no DP+PixelDP
- **Version3**: without using socket but with DP+PixelDP (requires more packages)

Other versions will be released soon.

Please cite the main paper if useful: [SplitFed Paper](https://arxiv.org/pdf/2004.12088.pdf)

---

## **Description**

This repository contains the implementation of:
- **Centralized Learning** (baseline)
- **Federated Learning**
- **Split Learning**
- **SplitFedV1 Learning**
- **SplitFedV2 Learning**

All programs are written in **Python 3.7.2** using the **PyTorch** library (**PyTorch 1.2.0**).

**Dataset**: HAM10000  
**Model**: ResNet18

---

## **Setup and Installation**

### **1. Clone the Repository**
```sh
git clone https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning.git
cd SplitFed-When-Federated-Learning-Meets-Split-Learning
```

### **2. Create a Virtual Environment**
It is recommended to create a virtual environment to manage dependencies.
```sh
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

### **3. Install Dependencies**
Since `torch==1.2.0` is not available on PyPI, use the closest version:
```sh
pip install torch==1.4.0 torchvision==0.5.0
pip install -r requirements.txt
```

---

## **Dataset Download and Preparation**

### **1. Download the HAM10000 Dataset**
The dataset can be downloaded from Kaggle:
- **[HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**
- Click **Download** → Extract the ZIP file

Alternatively, use the **Kaggle API**:
```sh
pip install kaggle  # Install Kaggle API
mkdir -p ~/.kaggle  # Create Kaggle directory
mv ~/Downloads/kaggle.json ~/.kaggle/  # Move API key
chmod 600 ~/.kaggle/kaggle.json  # Set permissions
```
Then, download the dataset:
```sh
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ./
unzip skin-cancer-mnist-ham10000.zip
```

### **2. Organize the Dataset**
Ensure the dataset is structured as follows:
```
SplitFed-When-Federated-Learning-Meets-Split-Learning/
│── HAM10000_metadata.csv
│── HAM10000_images_part_1/
│── HAM10000_images_part_2/
│── FL_ResNet_HAM10000.py
```

### **3. Verify Image Paths**
Run this script to ensure images are correctly mapped:
```python
import pandas as pd
import os
from glob import glob

df = pd.read_csv("HAM10000_metadata.csv")
imageid_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join("HAM10000_images_part_1", "*.jpg")) + glob(os.path.join("HAM10000_images_part_2", "*.jpg"))}
df["path"] = df["image_id"].map(imageid_path.get)
missing_images = df["path"].isnull().sum()
print(f"Missing images: {missing_images}")
```
If missing images > 0, recheck the dataset extraction.

---

## **Code Adjustments & Fixes**

### **1. Adjust Image Loading & Resizing**
Modify image resizing to **224x224** in your dataset class:
```python
X = Image.open(self.df['path'][index]).resize((224, 224))
```
Also update `transforms`:
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### **2. Fix Pooling Layer Issue**
Modify `avgpool` in `ResNet18` to prevent zero-size tensors:
```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```

### **3. Fix Fully Connected Layer Mismatch**
Modify `fc` layer in `ResNet18`:
```python
self.fc = nn.Linear(2048, num_classes)
```

### **4. Disable Multi-GPU if Necessary**
If you have only one GPU, disable `DataParallel` in `FL_ResNet_HAM10000.py`:
```python
# if torch.cuda.device_count() > 1:
#     print("We use", torch.cuda.device_count(), "GPUs")
#     net_glob = nn.DataParallel(net_glob)
```

---

## **Common Errors & Fixes**

| **Error** | **Cause** | **Fix** |
|-----------|----------|---------|
| `No module named 'sklearn'` | `scikit-learn` is missing | `pip install scikit-learn` |
| `Output size is too small` | Pooling layer reducing dimensions to (0x0) | 1. Change `self.avgpool = nn.AdaptiveAvgPool2d((1, 1))` 2.gdas|
| `size mismatch, m1: [64 x 2048], m2: [512 x 7]` | Incorrect FC layer input | Change `self.fc = nn.Linear(2048, num_classes)` |
| `Caught RuntimeError in replica 0 on device 0` | Multi-GPU issue | Disable `nn.DataParallel` |

---

## **Run Training**
Once all fixes are applied, run the script:
```sh
python FL_ResNet_HAM10000.py
```

If you encounter errors, check the **Common Errors & Fixes** section.

---

## **Conclusion**
This README documents the full setup process, dataset download, code adjustments, and error debugging steps. By following these instructions, you should be able to successfully implement and run **Federated Learning** and **Split Learning** using the **HAM10000 dataset** with **ResNet18**.

For any issues, feel free to open an issue in the GitHub repository! 🚀


