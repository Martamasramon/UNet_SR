import numpy as np
import torch
from torchvision import transforms as T
from skimage import transform
from PIL import Image
import random

class Resize(object):
  def __init__(self, output_size, down_factor, same_size_input_label=True):
    self.output_size  = output_size
    self.down_factor = down_factor
    self.same_size_input_label = same_size_input_label

  def __call__(self, sample):
    image = sample['lowres']

    h, w  = image.shape[:2]
    down_size = self.output_size // self.down_factor
    
    image = transform.resize(image, (down_size, down_size),preserve_range=True)
    if self.same_size_input_label:
      image = transform.resize(image, (self.output_size, self.output_size),preserve_range=True)

    sample['lowres'] = image
    return sample


class CenterCrop(object):
  def __init__(self, output_size):
    self.output_size = (output_size, output_size)

  def __call__(self, sample):
    image, label = sample['lowres'], sample['highres']

    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top  = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[top: top + new_h, left: left + new_w]
    label = label[top: top + new_h, left: left + new_w]

    sample['lowres']  = image
    sample['highres'] = label
    return sample


class ToTensor(object):
  def __call__(self, sample):
    image, label = sample['lowres'], sample['highres']

    if len(image.shape)==2:
      image = np.expand_dims(image, axis=0)
      label = np.expand_dims(label, axis=0)
    
    sample['lowres'] = torch.from_numpy(image).float()
    sample['highres'] = torch.from_numpy(label).float()
    return sample


class DownscaleBlurUpscale(object):
  def __init__(self, kernel_size=3, gaussian_sigma_max=0.1):
    self.gaussian_sigma_max = gaussian_sigma_max
    self.kernel_size        = kernel_size

  def __call__(self, sample):
    image = sample['lowres']
    image = image.squeeze(0).numpy()
    
    img_h, img_w = image.shape[:2]
    image = transform.resize(image, (img_h//2, img_w//2),preserve_range=True)    
    image = torch.FloatTensor(image).unsqueeze(0)

    gaussian_sigma = np.random.uniform(0.01,self.gaussian_sigma_max)
    image = T.GaussianBlur(self.kernel_size, gaussian_sigma)(image)
    image = image.squeeze(0).numpy()
    image = transform.resize(image, (img_h, img_w),preserve_range=True)
    
    sample['lowres'] = torch.from_numpy(image).unsqueeze(0).float()
    return sample
  
  
class RandomHorFlip(object):
  def __init__(self, p=0.5):
    self.p = p
    
  def __call__(self, sample):
    if random.random() > self.p:
      sample['lowres'] = np.fliplr(sample['lowres']).copy()
      sample['highres'] = np.fliplr(sample['highres']).copy()
    return sample


def get_train_transform(img_size=64, down_factor=2, same_size_input_label=True):
  return T.Compose([
      CenterCrop(img_size),
      Resize(img_size,down_factor=down_factor, same_size_input_label=same_size_input_label),
      RandomHorFlip(),
      ToTensor(),
      DownscaleBlurUpscale(3, 0.2)
  ])

def get_test_transform(img_size=64, down_factor=2, same_size_input_label=True):
  return T.Compose([
      CenterCrop(img_size),
      Resize(img_size,down_factor=down_factor, same_size_input_label=same_size_input_label),
      ToTensor(),
  ])

def get_t2w_transform(img_size=64):
  return T.Compose([
      T.CenterCrop(img_size*2),
      T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
      T.ToTensor()
  ]) 
  