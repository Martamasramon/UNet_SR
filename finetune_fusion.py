import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau

from dataset.dataset import MyDataset
from loss            import VGGPerceptualLoss, ssim_loss
from loss            import cosine_contrastive_loss as contrastive_loss

from model.train_functions    import train_evaluate, get_checkpoint_name, CHECKPOINTS_FOLDER
from model.runet              import RUNet_fusion

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

parser.add_argument('--checkpoint', type=str,   default='pretrain_PICAI')
parser.add_argument('--save_as',    type=str,   default=None)

parser.add_argument('--finetune',     action='store_true')
parser.set_defaults(finetune=False)
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
data_folder = 'HistoMRI' if args.finetune else 'PICAI'

train_dataset = MyDataset(
    folder + data_folder, 
    data_type       = 'train', 
    img_size        = args.img_size, 
    is_finetune     = args.finetune, 
    surgical_only   = args.surgical_only, 
    use_T2W         = args.use_T2W, 
)
test_dataset = MyDataset(
    folder + data_folder, 
    data_type       = 'test', 
    img_size        = args.img_size, 
    is_finetune     = args.finetune, 
    surgical_only   = args.surgical_only,
    use_T2W         = args.use_T2W, 
)
train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True)#,  num_workers=8)
test_dataloader  = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False)#, num_workers=0)

if args.checkpoint is None:
    checkpoint = args.save_as if args.save_as is not None else get_checkpoint_name()+'_fusion'
else:
    print('Loading model weights...')
    model.load_state_dict(torch.load(f'{CHECKPOINTS_FOLDER}{args.checkpoint}.pth'), strict=False)
    checkpoint = args.save_as if args.save_as is not None else args.checkpoint + '_fusion'
    
print('Starting training...')
losses = {'Pixel': L1Loss(),     'Perceptual': VGGPerceptualLoss(), 'SSIM': ssim_loss}
λ_loss = {'Pixel': args.λ_pixel, 'Perceptual': args.λ_perct,        'SSIM': args.λ_ssim}

optimizer = optim.Adam(model.parameters(), lr = args.lr)
scheduler = ReduceLROnPlateau(
    optimizer, 
    'min', 
    factor   = args.factor, 
    patience = args.patience, 
    cooldown = args.cooldown, 
    min_lr   = 1e-7
)
train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args.n_epochs, checkpoint, losses, λ_loss)