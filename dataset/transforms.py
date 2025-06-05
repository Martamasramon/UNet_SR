import numpy as np
import torch
from torchvision import transforms
from skimage import transform
from PIL import Image
import random

class Resize(object):
  def __init__(self, output_size, scale_factor, same_size_input_label=True):
    self.output_size  = output_size
    self.scale_factor = scale_factor
    self.same_size_input_label = same_size_input_label

  def __call__(self, sample):
    image = sample['image']

    h, w  = image.shape[:2]
    down_size = self.output_size // self.scale_factor
    
    image = transform.resize(image, (down_size, down_size),preserve_range=True)
    if self.same_size_input_label:
      image = transform.resize(image, (self.output_size, self.output_size),preserve_range=True)

    sample['image'] = image
    return sample


class CenterCrop(object):
  def __init__(self, output_size):
    self.output_size = (output_size, output_size)

  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top  = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[top: top + new_h, left: left + new_w]
    label = label[top: top + new_h, left: left + new_w]

    sample['image'] = image
    sample['label'] = label
    return sample


class ToTensor(object):
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    if len(image.shape)==2:
      image = np.expand_dims(image, axis=0)
      label = np.expand_dims(label, axis=0)
    
    sample['image'] = torch.from_numpy(image).float()
    sample['label'] = torch.from_numpy(label).float()
    return sample


class DownscaleBlurUpscale(object):
  def __init__(self, kernel_size=3, gaussian_sigma_max=0.1):
    self.gaussian_sigma_max = gaussian_sigma_max
    self.kernel_size        = kernel_size

  def __call__(self, sample):
    image = sample['image']
    image = image.squeeze(0).numpy()
    
    img_h, img_w = image.shape[:2]
    image = transform.resize(image, (img_h//2, img_w//2),preserve_range=True)    
    image = torch.FloatTensor(image).unsqueeze(0)

    gaussian_sigma = np.random.uniform(0.01,self.gaussian_sigma_max)
    image = transforms.GaussianBlur(self.kernel_size, gaussian_sigma)(image)
    image = image.squeeze(0).numpy()
    image = transform.resize(image, (img_h, img_w),preserve_range=True)
    
    sample['image'] = torch.from_numpy(image).unsqueeze(0).float()
    return sample
  
  
class RandomHorFlip(object):
  def __init__(self, p=0.5):
    self.p = p
    
  def __call__(self, sample):
    if random.random() > self.p:
      sample['image'] = np.fliplr(sample['image']).copy()
      sample['label'] = np.fliplr(sample['label']).copy()
    return sample


def create_transforms(img_size=64, scale_factor=4, same_size_input_label=True):
  train_transforms = transforms.Compose([
      CenterCrop(img_size),
      Resize(img_size,scale_factor=scale_factor, same_size_input_label=same_size_input_label),
      RandomHorFlip(),
      ToTensor(),
      DownscaleBlurUpscale(3, 0.2)
  ])
  test_transforms = transforms.Compose([
      CenterCrop(img_size),
      Resize(img_size,scale_factor=scale_factor, same_size_input_label=same_size_input_label),
      ToTensor(),
  ])
  return train_transforms, test_transforms

  