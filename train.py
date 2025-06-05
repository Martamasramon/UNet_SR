import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau

from dataset.dataset        import MyDataset
from dataset.transforms     import create_transforms
from loss           import VGGPerceptualLoss
from runet.runet    import RUNet
from runet.training_functions import train_evaluate
from utils.formatter import get_checkpoint_name

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size',       type=int,  default=64)
parser.add_argument('--scale_factor',   type=int,  default=2)
parser.add_argument('--train_bs',       type=int,  default=16)
parser.add_argument('--test_bs',        type=int,  default=1)
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)

parser.add_argument('--n_epochs',   type=int,   default=200)
parser.add_argument('--lr',         type=float, default=1e-5)
parser.add_argument('--factor',     type=float, default=0.5)
parser.add_argument('--patience',   type=int,   default=4)
parser.add_argument('--cooldown',   type=int,   default=2)

parser.add_argument('--λ_pixel',        type=float,  default=10.0)
parser.add_argument('--λ_perceptual',   type=float,  default=0.01)
parser.add_argument('--percpt_delay',   type=int,    default=50)

parser.add_argument('--img_folder',   type=str,  default='/cluster/project7/backup_masramon/IQT/PICAI/ADC/')
parser.add_argument('--is_pretrain',  action='store_true')
parser.add_argument('--finetune',     dest='is_pretrain', action='store_false')
parser.set_defaults(is_pretrain=True)
args, unparsed = parser.parse_known_args()
print('\n',args)

# Create model
print('\nCreating model...')
model = RUNet(args.drop_first, args.drop_last)
model = model.cuda()

# Create dataset & dataloader
print('Creating datasets...')
train_transforms, test_transforms = create_transforms(args.img_size, args.scale_factor)
train_dataset   = MyDataset(args.img_folder, train_transforms, is_pretrain=args.pretrain, is_train=True)
test_dataset    = MyDataset(args.img_folder, test_transforms,  is_pretrain=args.pretrain, is_train=False)

train_dataloader  = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=8)
test_dataloader   = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)

pixel_fn      = L1Loss()  # or MSELoss if preferred
perceptual_fn = VGGPerceptualLoss()

# Run training 
print('Starting training...')
checkpoint = get_checkpoint_name()

# 1. Train only with pixel loss
print('\n1. Train only with pixel loss...')
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.factor, patience=args.patience, cooldown=args.cooldown, min_lr=1e-7)
train_evaluate(model, train_dataloader, test_dataloader, pixel_fn, None,  args.λ_pixel, 0                , optimizer, scheduler, args.percpt_delay, checkpoint+'_stage_1')

# 2. Add perceptual loss
print('\n2. Train with perceptual loss...')

print('Loading best weights from stage 1...')
model.load_state_dict(torch.load(checkpoint + '_stage_1_best.pth'))

optimizer = optim.Adam(model.parameters(), lr=args.lr*0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.factor, patience=args.patience, cooldown=args.cooldown, min_lr=1e-7)
train_evaluate(model, train_dataloader, test_dataloader, pixel_fn, perceptual_fn, args.λ_pixel, args.λ_perceptual, optimizer, scheduler, args.n_epochs-args.percpt_delay, checkpoint+'_stage_2')

