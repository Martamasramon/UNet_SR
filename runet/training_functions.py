import os
import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import utils
from tqdm import tqdm
from utils.formatter import format_best_checkpoint_name, format_current_checkpoint_name
from torchvision.transforms import Normalize
from piq import ssim

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def transform_perceptual(img):
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.repeat(1, 3, 1, 1)
    img = transform(img)
    return img

# def show_imgs(images, titles=["Input", "Output", "Label"]):
#     plt.figure(figsize=(20,20))
#     for ind, img in enumerate(images):
#         plt.subplot(1, len(images), ind+1)
#         plt.imshow(utils.make_grid(images[ind]).cpu().numpy().transpose((1,2,0)))
#         plt.title(titles[ind])
#     plt.show()

def train(model, optimizer, dataloader, pixel_fn, perceptual_fn, λ_pixel, λ_perceptual):
    model.train()
    total_loss, pixel_total, perceptual_total, ssim_total = 0.0, 0.0, 0.0, 0.0
    
    for data in dataloader:
        # Get images
        images, labels = data["image"].float().cuda(), data["label"].float().cuda()

        optimizer.zero_grad()
        output = model(images)
        
        # Calculate loss
        pixel_loss = pixel_fn(output, labels)
        if perceptual_fn is not None:
            perceptual_loss = perceptual_fn(transform_perceptual(output), transform_perceptual(labels))
            loss = λ_pixel * pixel_loss + λ_perceptual * perceptual_loss
        else: 
            loss = pixel_loss
        ssim_loss = ssim(output, labels, data_range=1.0)
            
        loss.backward()
        optimizer.step()
        
        # Store losses
        total_loss  += loss.item()
        pixel_total += pixel_loss.item()
        ssim_total  += ssim_loss.item()    
        
        if perceptual_fn is not None:
            perceptual_total += perceptual_loss.item() 
        
    return total_loss / len(dataloader), pixel_total / len(dataloader), perceptual_total / len(dataloader), ssim_total / len(dataloader)

def evaluate(model, dataloader, pixel_fn, perceptual_fn, λ_pixel, λ_perceptual):
    model.eval()
    total_loss, pixel_total, perceptual_total, ssim_total = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for data in dataloader:
            # Get images
            images, labels = data["image"].float().cuda(), data["label"].float().cuda()
            output = model(images)

            # Calculate loss
            pixel_loss = pixel_fn(output, labels)
            if perceptual_fn is not None:
                perceptual_loss = perceptual_fn(transform_perceptual(output), transform_perceptual(labels))
                loss = λ_pixel * pixel_loss + λ_perceptual * perceptual_loss
            else: 
                loss = pixel_loss   
            ssim_loss = ssim(output, labels, data_range=1.0)

            # Store losses
            total_loss       += loss.item()
            pixel_total      += pixel_loss.item()
            ssim_total       += ssim_loss.item()      

            if perceptual_fn is not None:
                perceptual_total += perceptual_loss.item()

    return total_loss / len(dataloader), pixel_total / len(dataloader), perceptual_total / len(dataloader), ssim_total / len(dataloader)


def train_evaluate(model, train_dataloader, test_dataloader, pixel_fn, perceptual_fn, λ_pixel, λ_perceptual, optimizer, scheduler, n_epochs, name):
    best_loss = np.inf
    
    for epoch in range(n_epochs):
        print('\nEpoch ', epoch)
        print('Learning rate :', get_lr(optimizer))
        
        ###### Training ######
        avg_total, avg_pixel, avg_perceptual, val_ssim  = train(model, optimizer, train_dataloader, pixel_fn, perceptual_fn, λ_pixel, λ_perceptual)
        print(f"Training Loss: {avg_total:.4f} (Pixel: {avg_pixel:.4f}, Perceptual: {avg_perceptual:.4f}) - SSIM: {val_ssim:.4f}.")

        ###### Evaluation ######
        val_loss, val_pixel, val_perceptual, val_ssim = evaluate(model, test_dataloader, pixel_fn, perceptual_fn, λ_pixel, λ_perceptual)
        print(f"Validation Loss: {val_loss:.4f} (Pixel: {val_pixel:.4f}, Perceptual: {val_perceptual:.4f}) - SSIM: {val_ssim:.4f}.")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), name + '_best.pth')
            print('Best model saved at', name + '_best.pth')

        torch.save(model.state_dict(), name + '_current.pth')
        scheduler.step(val_loss)
        
    return model
