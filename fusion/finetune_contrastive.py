import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau

from dataset            import MyDataset
from train_functions    import train_evaluate, CHECKPOINTS_ADC, CHECKPOINTS_T2W

import sys
import os

sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet_SR'))
from loss    import ssim_loss #, VGGPerceptualLoss
from loss    import cosine_contrastive_loss as contrastive_loss
from model.runet  import RUNet

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

parser.add_argument('--n_epochs',   type=int,   default=50)
parser.add_argument('--lr',         type=float, default=1e-6)
parser.add_argument('--factor',     type=float, default=0.5)
parser.add_argument('--patience',   type=int,   default=4)
parser.add_argument('--cooldown',   type=int,   default=2)

parser.add_argument('--λ_pixel',    type=float, default=10.0)
parser.add_argument('--λ_perct',    type=float, default=0.01)
parser.add_argument('--λ_ssim',     type=float, default=1.0)
parser.add_argument('--λ_contrast', type=float, default=1.0)

parser.add_argument('--checkpoint_adc', type=str,   default='pretrain_PICAI')
parser.add_argument('--checkpoint_t2w', type=str,   default='default_64')

parser.add_argument('--finetune',     action='store_true')
parser.set_defaults(finetune=False)
parser.add_argument('--surgical_only',  action='store_true')
parser.set_defaults(surgical_only=False)
args, unparsed = parser.parse_known_args()

def get_scheduler(model, args):
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor   = args.factor, 
        patience = args.patience, 
        cooldown = args.cooldown, 
        min_lr   = 1e-8
    )
    return optimizer, scheduler
    
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