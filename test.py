import argparse
import torch
from dataset.dataset      import MyDataset
from model.runet          import RUNet, RUNet_768, RUNet_fusion
from model.test_functions import visualize_results, evaluate_results
from torch.utils.data     import DataLoader
from fusion.train_functions import CHECKPOINTS_ADC
from arguments              import args

import sys
import os
sys.path.append(os.path.abspath('/cluster/project7/ProsRegNet_CellCount/UNet/runet_t2w'))
from runetv2 import RUNet as T2Wnet

folder = '/cluster/project7/backup_masramon/IQT/'
        
def main():
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
    visualize_results(model, dataset, device, save_name, use_T2W=args.use_T2W, batch_size=args.test_bs)

    ## EVALUATE
    dataloader = DataLoader(dataset,  batch_size=args.test_bs,  shuffle=False)
    evaluate_results(model, dataloader, device, args.test_bs, use_T2W=args.use_T2W)


if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()