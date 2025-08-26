import argparse
import torch
from dataset.dataset      import MyDataset
from model.runet          import RUNet, RUNet_768, RUNet_fusion
from model.test_functions import visualize_results, evaluate_results
from torch.utils.data     import DataLoader
from fusion.train_functions import CHECKPOINTS_ADC

import sys
import os
sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet as T2Wnet

folder = '/cluster/project7/backup_masramon/IQT/'
        
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',     type=str,   default='pretrain_PICAI')
parser.add_argument('--checkpoint_t2w', type=str,   default='default_64_cont')

parser.add_argument('--img_size',     type=int,  default=64)
parser.add_argument('--drop_first',   type=float,default=0.1)
parser.add_argument('--drop_last',    type=float,default=0.5)
parser.add_argument('--batch_size',   type=int,  default=15)

parser.add_argument('--use_T2W',    action='store_true')
parser.set_defaults(use_T2W=False)
parser.add_argument('--fusion',     action='store_true')
parser.set_defaults(fusion=False)
parser.add_argument('--finetune',   action='store_true')
parser.set_defaults(finetune=False)
args, unparsed = parser.parse_known_args()
print('\n',args)

# Set device
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset
print('Creating dataset...')
data_folder = 'HistoMRI' if args.finetune else 'PICAI'

dataset = MyDataset(
    folder + data_folder, 
    data_type       = 'val', 
    img_size        = args.img_size, 
    is_finetune     = args.finetune, 
    use_T2W         = args.use_T2W, 
    t2w_model_path  = args.checkpoint_t2w
)

# Load models
print(f"Loading ADC-SR RUNet weights from {args.checkpoint}")
if args.fusion == True:
    model = RUNet_fusion(args.drop_first, args.drop_last).to(device)
    model.load_state_dict(torch.load(f"{CHECKPOINTS_ADC}/{args.checkpoint}.pth"))
else:
    try:
        model   = RUNet(args.drop_first, args.drop_last).to(device)
        model.load_state_dict(torch.load(f"{CHECKPOINTS_ADC}/{args.checkpoint}.pth"))
    except:
        print('Building RUNet v2 (embedding size 768)')
        model   = RUNet_768(args.drop_first, args.drop_last).to(device)
        model.load_state_dict(torch.load(f"{CHECKPOINTS_ADC}/{args.checkpoint}.pth"))

save_name = args.checkpoint+'_HistoMRI' if args.finetune else args.checkpoint+'_PICAI' 
visualize_results(model, dataset, device, save_name, use_T2W=args.use_T2W, batch_size=args.batch_size)

## EVALUATE
dataloader = DataLoader(dataset,  batch_size=args.batch_size,  shuffle=False)
evaluate_results(model, dataloader, device, args.batch_size, use_T2W=args.use_T2W)
