import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
import copy
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt

# Set Random Seed for Reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Program Identification
program = "FL ResNet18 on CIFAR-10"
print(f"---------{program}----------")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print Colored Logs
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

# Hyperparameters
num_users = 5
epochs = 200
frac = 1
lr = 0.0001

# =====================================================================================
#                            Load CIFAR-10 Dataset
# =====================================================================================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 Mean & Std
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# =====================================================================================
#                            Split Data Among Clients (IID)
# =====================================================================================
def dataset_iid(dataset, num_users):
    num_items = len(dataset) // num_users
    dict_users = {}
    all_idxs = list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    

dict_users = dataset_iid(trainset, num_users)
dict_users_test = dataset_iid(testset, num_users)

# =====================================================================================
#                            Client-Side Data Splitting
# =====================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        image, label = self.dataset[self.idxs[idx]]
        return image, label

# =====================================================================================
#                             Client Training & Evaluation
# =====================================================================================
class LocalUpdate(object):
    def __init__(self, idx, lr, device, dataset_train, dataset_test, idxs, idxs_test):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=False)

    def train(self, net):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        epoch_acc, epoch_loss = [], []

        for iter in range(self.local_ep):
            batch_acc, batch_loss = [], []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                fx = net(images)
                loss = self.loss_func(fx, labels)
                acc = (fx.argmax(dim=1) == labels).float().mean().item()
                
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
                batch_acc.append(acc)
            
            prRed(f'Client {self.idx} Train => Local Epoch: {iter} | Acc: {acc:.3f} | Loss: {loss.item():.4f}')
            epoch_loss.append(np.mean(batch_loss))
            epoch_acc.append(np.mean(batch_acc))
        return net.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)
    
    def evaluate(self, net):
        net.eval()
        epoch_acc, epoch_loss = [], []

        with torch.no_grad():
            batch_acc, batch_loss = [], []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                
                fx = net(images)
                loss = self.loss_func(fx, labels)
                acc = (fx.argmax(dim=1) == labels).float().mean().item()
                
                batch_loss.append(loss.item())
                batch_acc.append(acc)
            
            prGreen(f'Client {self.idx} Test => Acc: {acc:.3f} | Loss: {loss.item():.4f}')
            epoch_loss.append(np.mean(batch_loss))
            epoch_acc.append(np.mean(batch_acc))
        return np.mean(epoch_loss), np.mean(epoch_acc)

# =====================================================================================
#                            ResNet18 Model Definition
# =====================================================================================
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.resnet.fc = nn.Linear(512, num_classes)  # Modify FC layer for CIFAR-10

    def forward(self, x):
        return self.resnet(x)

net_glob = ResNet18(num_classes=10).to(device)

# =====================================================================================
#                            Federated Learning Training
# =====================================================================================
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

net_glob.train()
w_glob = net_glob.state_dict()

loss_train_collect, acc_train_collect = [], []
loss_test_collect, acc_test_collect = [], []

for iter in range(epochs):
    w_locals, loss_locals_train, acc_locals_train = [], [], []
    loss_locals_test, acc_locals_test = [], []
    
    idxs_users = np.random.choice(range(num_users), max(int(frac * num_users), 1), replace=False)
    
    for idx in idxs_users:
        local = LocalUpdate(idx, lr, device, trainset, testset, dict_users[idx], dict_users_test[idx])
        w, loss_train, acc_train = local.train(copy.deepcopy(net_glob).to(device))
        loss_test, acc_test = local.evaluate(copy.deepcopy(net_glob).to(device))
        
        w_locals.append(copy.deepcopy(w))
        loss_locals_train.append(loss_train)
        acc_locals_train.append(acc_train)
        loss_locals_test.append(loss_test)
        acc_locals_test.append(acc_test)

    w_glob = FedAvg(w_locals)
    net_glob.load_state_dict(w_glob)

    acc_train_collect.append(np.mean(acc_locals_train))
    acc_test_collect.append(np.mean(acc_locals_test))

print("Training and Evaluation completed!")

df = DataFrame({'round': list(range(1, len(acc_train_collect)+1)), 'acc_train': acc_train_collect, 'acc_test': acc_test_collect})
df.to_excel("FL_CIFAR10.xlsx", index=False)
