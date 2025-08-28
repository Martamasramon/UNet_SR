import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader

from fusion_train_functions    import train_evaluate, get_scheduler, CHECKPOINTS_ADC, CHECKPOINTS_T2W

import sys
import os

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet_SR'))
from loss           import ssim_loss #, VGGPerceptualLoss
from loss           import cosine_contrastive_loss as contrastive_loss
from model.runet    import RUNet
from arguments      import args
from dataset.dataset import MyDataset

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet as T2Wnet

folder = '/cluster/project7/backup_masramon/IQT/'

    
def main():
    # Create model
    print('\nCreating models...')
    model_adc = RUNet(args.drop_first, args.drop_last).cuda()
    model_t2w = T2Wnet(0.1, 0.5, img_size=args.img_size).cuda()
    
    # Load models
    print('Loading model weights...')
    model_adc.load_state_dict(torch.load(f'{CHECKPOINTS_ADC}{args.checkpoint_adc}.pth'))
    model_t2w.load_state_dict(torch.load(f'{CHECKPOINTS_T2W}{args.checkpoint_t2w}.pth'))
    
    checkpoint_adc = args.checkpoint_adc + '_cont'
    checkpoint_t2w = args.checkpoint_t2w + '_cont'

    # Create dataset & dataloader
    print('Creating datasets...')
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'

    train_dataset = MyDataset(
        folder + data_folder, 
        data_type       = 'train', 
        img_size        = args.img_size, 
        is_finetune     = args.finetune, 
        use_T2W         = True, 
        t2w_model_path  = args.checkpoint_t2w
    )
    test_dataset = MyDataset(
        folder + data_folder, 
        data_type       = 'test', 
        img_size        = args.img_size, 
        is_finetune     = args.finetune, 
        use_T2W         = True, 
        t2w_model_path  = args.checkpoint_t2w
    )
    train_dl = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=8)
    test_dl  = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)

    # Train        
    print('Starting training...')
    losses = {'Pixel': L1Loss(),     'SSIM': ssim_loss,   'Contrastive': contrastive_loss} #'Perceptual': VGGPerceptualLoss(),
    λ_loss = {'Pixel': args.λ_pixel, 'SSIM': args.λ_ssim, 'Contrastive': args.λ_contrast}  #'Perceptual': args.λ_perct,

    optim_adc, sched_adc = get_scheduler(model_adc, args)
    optim_t2w, sched_t2w = get_scheduler(model_t2w, args)
   
    train_evaluate(model_adc, model_t2w, train_dl, test_dl, optim_adc, optim_t2w, sched_adc, sched_t2w, args.n_epochs, checkpoint_adc, checkpoint_t2w, losses, λ_loss)


if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()