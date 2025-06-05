from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

import matplotlib.pyplot as plt

class MyDataset(Dataset):
  def __init__(self, img_path, transforms, use_histo=False, use_t2w=False, is_pretrain=True, is_train=True):
    root   = 'pretrain' if is_pretrain else 'finetune'
    suffix = 'train'    if is_train    else 'test'
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'../Dataset_preparation/{root}_{suffix}.csv')
    self.transforms = transforms
    self.use_histo  = use_histo
    self.use_t2w    = use_t2w

  def __len__(self):
    return len(self.img_dict)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(self.img_path + item["SID"])
    img    = np.array(img)/255.0
    sample = {'image': img, 'label': img}
    
    if self.use_histo:
      embed_histo = torch.load(item["Histo"],weights_only=True) # torch.load(map_location=torch.device("cpu")
      sample['Histo'] = embed_histo
    
    if self.use_t2w:
      embed_t2w = torch.load(item["T2W"],weights_only=True) # torch.load(map_location=torch.device("cpu")
      sample['T2W'] = embed_t2w

    if self.transforms:
      sample = self.transforms(sample)
      
    # ### Test by plotting
    # img = np.squeeze(sample['image'].cpu().detach().numpy())
    # lbl = np.squeeze(sample['label'].cpu().detach().numpy())
      
    # _, ax = plt.subplots(1, 2, figsize=(8, 4))
    # im0 = ax[0].imshow(img, cmap='gray')
    # plt.colorbar(im0, ax=ax[0])
    # im1 = ax[1].imshow(lbl, cmap='gray')
    # plt.colorbar(im1, ax=ax[1])
    # plt.tight_layout()
    # plt.savefig(f'image_{idx}.jpg')
    # plt.close()
            
    return sample

  def __len__(self):
    return len(self.img_dict)
