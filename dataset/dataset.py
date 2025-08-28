import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from dataset.transforms import get_train_transform, get_test_transform, get_t2w_transform

import sys
import os
sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet_SR/fusion'))
from fusion.fusion_train_functions import CHECKPOINTS_ADC, CHECKPOINTS_T2W

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet as T2Wnet

class MyDataset(Dataset):
  def __init__(
      self, 
      img_path, 
      data_type,
      img_size       = 64, 
      down_factor    = 2,
      use_T2W        = False, 
      is_finetune    = False, 
      surgical_only  = False,
      use_mask       = False
  ):
    root   = 'finetune' if is_finetune else 'pretrain'
    if surgical_only:
            root += '_surgical'
    
    self.masked     = '_mask' if use_mask else ''
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'/cluster/project7/ProsRegNet_CellCount/Dataset_preparation/{root}{self.masked}_{data_type}.csv')
    self.transform  = get_train_transform(img_size, down_factor) if data_type=='train' else get_test_transform(img_size, down_factor)
    self.t2w_transform = get_t2w_transform(img_size)
    self.use_T2W    = use_T2W
    
    print(f'Loading data from {root}{self.masked}_{data_type}...')

  def __len__(self):
    return len(self.img_dict)
  
  # def get_t2w_embedding(self, path):
  #   img       = Image.open(f'{self.img_path}/T2W{self.masked}/{path}').convert('L')
  #   input_img = self.t2w_transform(img).unsqueeze(0)
    
  #   with torch.no_grad():
  #     return self.t2w_model.get_embedding(input_img)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(f'{self.img_path}/ADC{self.masked}/{item["SID"]}').convert('L')
    img    = np.array(img)/255.0
    sample = self.transform({'lowres': img, 'highres': img})
    
    if self.use_T2W:
      t2w = Image.open(f'{self.img_path}/T2W{self.masked}/{item["SID"]}').convert('L')      
      sample['T2W'] = self.t2w_transform(t2w)
    
    return sample

  def __len__(self):
    return len(self.img_dict)
