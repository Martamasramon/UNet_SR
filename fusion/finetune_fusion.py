import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader

from train_functions    import train_evaluate_adc, get_scheduler, CHECKPOINTS_ADC, CHECKPOINTS_T2W

import sys
import os

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet_SR'))
from loss           import ssim_loss #, VGGPerceptualLoss
from loss           import cosine_contrastive_loss as contrastive_loss
from model.runet    import RUNet_fusion 
from arguments      import args
from dataset.dataset import MyDataset

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet as T2Wnet

folder = '/cluster/project7/backup_masramon/IQT/'


def freeze_encoder(model):
    for name, param in model.named_parameters():
        if 'block' in name or 'representation_transform' in name:
            param.requires_grad = False
    return model
            
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def main():
    # Create model
    print('\nCreating models...')
    model_adc = RUNet_fusion(args.drop_first, args.drop_last).cuda()
    model_t2w = T2Wnet(args.drop_first, args.drop_last, img_size=args.img_size).cuda()
    
    # Load models
    print('Loading model weights...')
    model_adc.load_state_dict(torch.load(f'{CHECKPOINTS_ADC}{args.checkpoint_adc}.pth'), strict=False)
    model_t2w.load_state_dict(torch.load(f'{CHECKPOINTS_T2W}{args.checkpoint_t2w}.pth'))
    model_t2w.eval()
    
    checkpoint_adc = args.checkpoint_adc + '_fusion'

    # Create dataset & dataloader
    print('Creating datasets...')
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'

    train_dataset = MyDataset(
        folder + data_folder, 
        data_type       = 'train', 
        img_size        = args.img_size, 
        is_finetune     = args.finetune, 
        use_T2W         = True, 
    )
    test_dataset = MyDataset(
        folder + data_folder, 
        data_type       = 'test', 
        img_size        = args.img_size, 
        is_finetune     = args.finetune, 
        use_T2W         = True, 
    )
    train_dl = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=8)
    test_dl  = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)
    
    losses = {'Pixel': L1Loss(),     'SSIM': ssim_loss,   'Contrastive': contrastive_loss} #'Perceptual': VGGPerceptualLoss(),
    λ_loss = {'Pixel': args.λ_pixel, 'SSIM': args.λ_ssim, 'Contrastive': args.λ_contrast}  #'Perceptual': args.λ_perct,
   
    # Train with frozen encoder
    print('Starting training stage 1 (frozen encoder)...')
    model_adc            = freeze_encoder(model_adc)
    optim_adc, sched_adc = get_scheduler(model_adc, args, args.lr_1)
    train_evaluate_adc(model_adc, model_t2w, train_dl, test_dl, optim_adc, sched_adc, args.n_epochs_1, checkpoint_adc + '_1', losses, λ_loss)

    # Finetune encoder
    print('Starting training stage 2 (unfrozen)...')
    model_adc.load_state_dict(torch.load(f'{CHECKPOINTS_ADC}{args.checkpoint_adc}_fusion_1_best.pth'))
    model_adc            = unfreeze(model_adc)
    optim_adc, sched_adc = get_scheduler(model_adc, args, args.lr_2)
    train_evaluate_adc(model_adc, model_t2w, train_dl, test_dl, optim_adc, sched_adc, args.n_epochs_2, checkpoint_adc + '_2', losses, λ_loss)

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()