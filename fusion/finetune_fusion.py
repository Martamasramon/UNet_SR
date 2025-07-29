import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau

from dataset            import MyDataset
from train_functions    import train_evaluate_adc, CHECKPOINTS_ADC, CHECKPOINTS_T2W

import sys
import os

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet_SR'))
from loss    import ssim_loss #, VGGPerceptualLoss
from loss    import cosine_contrastive_loss as contrastive_loss
from model.runet  import RUNet_fusion 

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet as T2Wnet

folder = '/cluster/project7/backup_masramon/IQT/'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size',       type=int,  default=64)
parser.add_argument('--train_bs',       type=int,  default=16)
parser.add_argument('--test_bs',        type=int,  default=1)
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)

parser.add_argument('--use_T2W',    action='store_true')
parser.set_defaults(use_T2W=False)
parser.add_argument('--use_histo',  action='store_true')
parser.set_defaults(use_histo=False)

parser.add_argument('--n_epochs_1', type=int,   default=50)
parser.add_argument('--n_epochs_2', type=int,   default=50)
parser.add_argument('--lr_1',       type=float, default=1e-4)
parser.add_argument('--lr_2',       type=float, default=1e-6)
parser.add_argument('--factor',     type=float, default=0.5)
parser.add_argument('--patience',   type=int,   default=4)
parser.add_argument('--cooldown',   type=int,   default=2)

parser.add_argument('--λ_pixel',    type=float, default=10.0)
parser.add_argument('--λ_perct',    type=float, default=0.01)
parser.add_argument('--λ_ssim',     type=float, default=1.0)
parser.add_argument('--λ_contrast', type=float, default=1.0)

parser.add_argument('--checkpoint_adc', type=str,   default='pretrain_PICAI_cont')
parser.add_argument('--checkpoint_t2w', type=str,   default='default_64_cont')

parser.add_argument('--finetune',     action='store_true')
parser.set_defaults(finetune=False)
parser.add_argument('--surgical_only',  action='store_true')
parser.set_defaults(surgical_only=False)
args, unparsed = parser.parse_known_args()

def get_scheduler(model, args, lr):
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor   = args.factor, 
        patience = args.patience, 
        cooldown = args.cooldown, 
        min_lr   = 1e-8
    )
    return optimizer, scheduler

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
        use_t2w         = True, 
    )
    test_dataset = MyDataset(
        folder + data_folder, 
        data_type       = 'test', 
        img_size        = args.img_size, 
        is_finetune     = args.finetune, 
        use_t2w         = True, 
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