import torch
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error      as mse_metric

def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    
    mse  = mse_metric (gt_np, pred_np)
    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0)
    return mse, psnr, ssim

def evaluate_results(model, dataloader, device, batch_size):
    mse_list, psnr_list, ssim_list = [], [], []
    for batch in dataloader:
        lowres    = batch['lowres'].to(device)
        highres   = batch['highres'].to(device)
        
        with torch.no_grad():
            pred,_ = model(lowres)
        
        for j in range(pred.size(0)):
            mse, psnr, ssim = compute_metrics(pred[j], highres[j])
            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    print(f'Average MSE:  {np.mean(mse_list):.6f}')
    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')

def format_image(img):
    return np.squeeze((img).cpu().numpy())

def visualize_results(model, dataset, device, name, t2w=False, batch_size=5, seed=1):
    ncols = 4 if t2w else 3
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('Super resolution (Output)')
    axes[0,2].set_title('High res (Ground truth)')
    if t2w:
        axes[0,3].set_title('High res T2W')
        
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(dataset)),batch_size,replace=False)

    model.eval()
    with torch.no_grad():
        for i, ind in enumerate(indices):
            sample = dataset[ind]
            lowres    = sample['lowres'].unsqueeze(0).float().to(device)
            highres   = sample['highres'].unsqueeze(0).float().to(device)
            if t2w:
                t2w_image = sample['T2W'].to(device)
                
            # Use model to get prediction
            pred,_ = model(lowres)

            im0 = axes[i, 0].imshow(format_image(lowres),  cmap='gray', vmin=0, vmax=1)
            axes[i, 0].axis('off')
            fig.colorbar(im0, ax=axes[i, 0])

            im1 = axes[i, 1].imshow(format_image(pred),    cmap='gray', vmin=0, vmax=1)
            axes[i, 1].axis('off')
            fig.colorbar(im1, ax=axes[i, 1])
            
            im2 = axes[i, 2].imshow(format_image(highres), cmap='gray', vmin=0, vmax=1)
            axes[i, 2].axis('off')
            fig.colorbar(im2, ax=axes[i, 2])
            
            if t2w:
                im3 = axes[i, 3].imshow(format_image(t2w_image), cmap='gray', vmin=0, vmax=1)
                axes[i, 3].axis('off')
                fig.colorbar(im3, ax=axes[i, 3])
            
        fig.tight_layout(pad=0.25)
        plt.savefig(f'./results/image_{name}.jpg')
        plt.close()