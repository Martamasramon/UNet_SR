import argparse
import torch
from dataset.dataset      import MyDataset
from model.runet          import RUNet, RUNetv2
from model.test_functions import visualize_results, evaluate_results
from model.train_functions import CHECKPOINTS_FOLDER
from torch.utils.data     import DataLoader

folder = '/cluster/project7/backup_masramon/IQT/'
        
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   type=str,  default='')
parser.add_argument('--img_size',     type=int,  default=64)
parser.add_argument('--drop_first',   type=float,default=0.1)
parser.add_argument('--drop_last',    type=float,default=0.5)
parser.add_argument('--batch_size',   type=int,  default=15)

parser.add_argument('--use_T2W',    action='store_true')
parser.set_defaults(use_T2W=False)
parser.add_argument('--use_histo',  action='store_true')
parser.set_defaults(use_histo=False)

parser.add_argument('--finetune',     action='store_true')
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
    use_histo       = args.use_histo, 
    use_t2w         = args.use_T2W, 
)

print(f"Loading RUNet weights from {args.checkpoint}")
try:
    model   = RUNet(args.drop_first, args.drop_last).to(device)
    model.load_state_dict(torch.load(f"{CHECKPOINTS_FOLDER}/{args.checkpoint}.pth"))
except:
    print('Building RUNet v2 (embedding size 768)')
    model   = RUNetv2(args.drop_first, args.drop_last).to(device)
    model.load_state_dict(torch.load(f"{CHECKPOINTS_FOLDER}/{args.checkpoint}.pth"))

save_name = args.checkpoint+'_HistoMRI' if args.finetune else args.checkpoint+'_PICAI' 
visualize_results(model, dataset, device, save_name, t2w=args.use_T2W, batch_size=args.batch_size)

## EVALUATE
dataloader = DataLoader(dataset,  batch_size=args.batch_size,  shuffle=False)
evaluate_results(model, dataloader, device, args.batch_size)
