import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from transformers import ViTModel
from torchvision.models import vit_b_16
import albumentations as A
import torch.nn as nn

def read_img(path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb

class CropDataset(Dataset):
    def __init__(self, root):
        self.paths = glob(root)
        self.T = A.Compose([
            A.Normalize(),
            A.CenterCrop(512, 512),
            A.Resize(384, 384),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = read_img(path)
        img = self.T(image=img)['image']
        return img

# embedding train imgs
train_dataset = CropDataset('train/*/*/*')
test_dataset = CropDataset('test/*/*')
train_loader = DataLoader(train_dataset, batch_size=128, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8, pin_memory=True)

device = 'cuda:0'
model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1')
model.heads.head = nn.Identity()
model = model.to(device)
model.eval()

top = 3
topK_test = torch.empty(0)
topKscore_test = torch.empty(0)
with torch.no_grad():
    for i, test_inputs in enumerate(tqdm(test_loader)):
        test_outputs = model(test_inputs.to(device))
        Ct = torch.empty(0)
        for j, train_inputs in enumerate(tqdm(train_loader)):
            train_outputs = model(train_inputs.to(device))
            C = F.normalize(test_outputs) @ F.normalize(train_outputs).t()
            if len(Ct) == 0:
                Ct = C
            else:
                Ct = torch.cat((Ct, C), dim=1)
        topKscore = torch.topk(Ct, k=top, dim=1)[0]
        topK = torch.topk(Ct, k=top, dim=1)[1]
        if len(topK_test) == 0:
            topK_test = topK
            topKscore_test = topKscore
        else:
            topK_test = torch.cat((topK_test, topK), dim=0)
            topKscore_test = torch.cat((topKscore_test, topKscore), dim=0)

topK_test = topK_test.cpu().numpy()
topKscore_test = topKscore_test.cpu().numpy()
labels = np.array([int(path.split('/')[2]) for path in glob('train/*/*/*')])
paths = [path for path in glob('test/*/*')]
topK_labels = [[path] + labels[v].tolist() for path, v in zip(paths, topK_test)]
topKscore_labels = [[path] + v.tolist() for path, v in zip(paths, topKscore_test)]
sim_test = pd.DataFrame(topK_labels, columns=(['path'] + [i+1 for i in range(top)]))
sim_test_score = pd.DataFrame(topKscore_labels, columns=(['path'] + [i+1 for i in range(top)]))
sim_test.to_csv('test_sim_vit_512_384_tv_noflip.csv', index=False)
sim_test_score.to_csv('test_sim_vit_512_384_tv_noflip_score.csv', index=False)
        



