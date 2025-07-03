import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau

from dataset.dataset        import MyDataset
from loss                   import VGGPerceptualLoss, ssim_loss

from model.training_functions import train_evaluate, get_checkpoint_name
from model.runet              import RUNet

folder = '/cluster/project7/backup_masramon/IQT/'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size',       type=int,  default=64)
parser.add_argument('--train_bs',       type=int,  default=16)
parser.add_argument('--test_bs',        type=int,  default=1)
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)

parser.add_argument('--n_epochs_1', type=int,   default=100)
parser.add_argument('--lr',         type=float, default=1e-5)
parser.add_argument('--factor',     type=float, default=0.5)
parser.add_argument('--patience',   type=int,   default=3)
parser.add_argument('--cooldown',   type=int,   default=1)

parser.add_argument('--n_epochs_2',     type=int,    default=100)
parser.add_argument('--λ_pixel',        type=float,  default=1.0)
parser.add_argument('--λ_perct',        type=float,  default=0.)
parser.add_argument('--λ_ssim',         type=float,  default=0.)
parser.add_argument('--lr_factor',      type=float,  default=0.1)

parser.add_argument('--load_checkpoint',type=str,  default=None)
parser.add_argument('--img_folder',     type=str,  default='PICAI')
parser.add_argument('--is_pretrain',    action='store_true')
parser.set_defaults(is_pretrain=True)
parser.add_argument('--finetune',       dest='is_pretrain', action='store_false')
parser.add_argument('--surgical_only',  action='store_true')
parser.set_defaults(surgical_only=False)
args, unparsed = parser.parse_known_args()

print('Parameters:')
for key, value in vars(args).items():
    print(f'- {key}: {value}')
print('')

# Create model
print('\nCreating model...')
model = RUNet(args.drop_first, args.drop_last)
model = model.cuda()

# Create dataset & dataloader
print('Creating datasets...')
train_dataset = MyDataset(
    folder + args.img_folder, 
    img_size        = args.img_size, 
    is_pretrain     = args.is_pretrain, 
    surgical_only   = args.surgical_only, 
    is_train        = True
)
test_dataset = MyDataset(
    folder + args.img_folder, 
    img_size        = args.img_size, 
    is_pretrain     = args.is_pretrain, 
    surgical_only   = args.surgical_only, 
    is_train        = False
)
train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=8)
test_dataloader  = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)

# Run training 
print('Starting training...')
losses = {'Pixel': L1Loss(), 'Perceptual': VGGPerceptualLoss(), 'SSIM': ssim_loss}

checkpoint = get_checkpoint_name()


if args.load_checkpoint is None:
    # 1. Train only with pixel loss
    print('\n1. Train only with pixel loss...')
    λ_loss = {'Pixel': args.λ_pixel, 'Perceptual': 0, 'SSIM': 0}

    checkpoint = get_checkpoint_name()
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor   = args.factor, 
        patience = args.patience, 
        cooldown = args.cooldown, 
        min_lr   = 1e-7
    )
    train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args.n_epochs_1, checkpoint+'_stage_1', losses, λ_loss)
else:
    checkpoint = './checkpoints/' + args.load_checkpoint

# 2. Add perceptual loss
print('\n2. Train with perceptual loss...')
λ_loss = {'Pixel': args.λ_pixel, 'Perceptual': args.λ_perct, 'SSIM': args.λ_ssim }
print(λ_loss)

print('Loading best weights from stage 1...')
model.load_state_dict(torch.load(checkpoint + '_stage_1_best.pth'))

optimizer = optim.Adam(model.parameters(), lr=args.lr*args.lr_factor)
scheduler = ReduceLROnPlateau(
    optimizer, 
    'min', 
    factor   = args.factor, 
    patience = args.patience, 
    cooldown = args.cooldown, 
    min_lr   = 1e-7
)
train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args.n_epochs_2, checkpoint+'_stage_2', losses, λ_loss)