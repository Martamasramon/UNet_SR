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

def evaluate_results(model, model_t2w, dataloader, device, batch_size, fusion):
    mse_list, psnr_list, ssim_list = [], [], []
    
    for batch in dataloader:
        lowres    = batch['lowres'].to(device)
        highres   = batch['highres'].to(device)
        ###  FIX THIS
        
        with torch.no_grad():
            if fusion:
                _, embed = model_t2w(batch['T2W'].float().cuda())
                pred,_   = model(lowres, torch.squeeze(embed.to(device)))
            else:
                pred,_   = model(lowres)
        
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

def plot_image(image, fig, axes, i, j, colorbar=True):
    image  = format_image(image)
            
    img_plot = axes[i, j].imshow(image,  cmap='gray', vmin=0, vmax=1)
    axes[i, j].axis('off')
    if colorbar:
        fig.colorbar(img_plot, ax=axes[i, j])
    
def visualize_results(model, model_t2w, dataset, device, name, use_T2W, batch_size, fusion, seed=1):
    ncols = 5 if use_T2W else 4
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols, figsize=(3*ncols,3*batch_size))
    axes[0,0].set_title('Low res (Input)')
    axes[0,1].set_title('Super resolution (Output)')
    axes[0,2].set_title('Error')
    axes[0,3].set_title('High res (Ground truth)')
    if use_T2W:
        axes[0,4].set_title('High res T2W')
        
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(dataset)),batch_size,replace=False)

    model.eval()
    with torch.no_grad():
        for i, ind in enumerate(indices):
            sample = dataset[ind]
            lowres    = sample['lowres'].unsqueeze(0).float().to(device)
            highres   = sample['highres'].unsqueeze(0).float().to(device)
            
            # Use model to get prediction
            if use_T2W:
                t2w_image = sample['T2W'].to(device)
            if fusion:
                _, t2w_embed = model_t2w(t2w_image.unsqueeze(0).float().cuda())
                pred, _   = model(lowres, t2w_embed)
            else:
                pred,_ = model(lowres)
            
            # Plot images
            plot_image(lowres,  fig, axes, i, 0)
            plot_image(pred,    fig, axes, i, 1)
            plot_image(pred,    fig, axes, i, 2, False)
            plot_image(highres, fig, axes, i, 3)
            if use_T2W:
                plot_image(t2w_image, fig, axes, i, 4)

            # Error
            err = np.abs(format_image(pred) - format_image(highres))
            p99 = np.percentile(err, 99.5)
            den = p99 if p99 > 1e-8 else (err.max() + 1e-8)
            err_norm = np.clip(err / den, 0, 1)

            im_overlay = axes[i, 2].imshow(err_norm, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.6)
            cbar = fig.colorbar(im_overlay, ax=axes[i, 2])
            
            
        fig.tight_layout(pad=0.25)
        plt.savefig(f'./results/image_{name}.jpg')
        plt.close()