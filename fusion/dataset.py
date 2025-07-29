import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet_SR/dataset'))
from transforms  import get_train_transform, get_test_transform, get_t2w_transform

class MyDataset(Dataset):
  def __init__(
      self, 
      img_path, 
      data_type,
      img_size       = 64, 
      use_histo      = False, 
      use_t2w        = False, 
      is_finetune    = False, 
      surgical_only  = False,
      t2w_model_drop = [0.1,0.5],
      t2w_model_path = None,
      use_mask       = False
  ):
    root   = 'finetune' if is_finetune else 'pretrain'
    if surgical_only:
            root += '_surgical'
    
    self.masked     = '_mask' if use_mask else ''
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'/cluster/project7/ProsRegNet_CellCount/Dataset_preparation/{root}{self.masked}_{data_type}.csv')
    self.transform  = get_train_transform(img_size) if data_type=='train' else get_test_transform(img_size)
    self.t2w_transform = get_t2w_transform(img_size)
    self.use_histo  = use_histo
    self.use_t2w    = use_t2w
    
    print(f'Loading data from {root}{self.masked}_{data_type}...')
    
    if t2w_model_path is not None:
      # Load pre-trained T2W embedding model
      self.t2w_model = T2Wnet(t2w_model_drop[0], t2w_model_drop[1], img_size=img_size).cuda()
      self.t2w_model.load_state_dict(torch.load(t2w_model_path))
      self.t2w_model.eval() 
      

  def __len__(self):
    return len(self.img_dict)
  
  def get_t2w_embedding(self, path):
        img       = Image.open(f'{self.img_path}/T2W{self.masked}/{path}').convert('L')
        input_img = self.t2w_transform(img).unsqueeze(0)
        
        with torch.no_grad():
          return self.t2w_model.get_embedding(input_img)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(f'{self.img_path}/ADC{self.masked}/{item["SID"]}').convert('L')
    img    = np.array(img)/255.0
    sample = self.transform({'lowres': img, 'highres': img})
    
    if self.use_histo:
      sample['Histo'] = torch.load(item["Histo"],weights_only=True) 
    
    if self.use_t2w:
      t2w = Image.open(f'{self.img_path}/T2W{self.masked}/{item["SID"]}').convert('L')      
      sample['T2W'] = self.t2w_transform(t2w)
            
    return sample

  def __len__(self):
    return len(self.img_dict)
