import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader

from dataset.dataset        import MyDataset
from loss                   import VGGPerceptualLoss, ssim_loss

from model.train_functions  import train_evaluate, get_checkpoint_name, CHECKPOINTS_FOLDER, get_scheduler
from model.runet            import RUNet
from arguments              import args

def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    print('\nCreating model...')
    model = RUNet(args.drop_first, args.drop_last)
    model = model.to(device)

    # Create dataset & dataloader
    print('Creating datasets...')
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'

    train_dataset = MyDataset(
        data_folder, 
        data_type       = 'train', 
        img_size        = args.img_size, 
        down_factor     = args.down,
        is_finetune     = args.finetune, 
        use_mask        = args.masked
    )
    test_dataset = MyDataset(
        data_folder, 
        data_type       = 'test', 
        img_size        = args.img_size, 
        down_factor     = args.down,
        is_finetune     = args.finetune, 
        use_mask        = args.masked
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=8)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)

    # Run training 
    print('Starting training...')
    losses = {'Pixel': L1Loss(), 'Perceptual': VGGPerceptualLoss(), 'SSIM': ssim_loss}

    if args.checkpoint is None:
        # 1. Train only with pixel loss
        print('\n1. Train only with pixel loss...')
        λ_loss = {'Pixel': args.λ_pixel, 'Perceptual': 0, 'SSIM': 0}

        checkpoint = args.save_as if args.save_as is not None else get_checkpoint_name()
        optimizer, scheduler = get_scheduler(model, args) 
        train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args.n_epochs, checkpoint+'_stage_1', losses, λ_loss)
    else:
        checkpoint = args.save_as if args.save_as is not None else args.checkpoint

    # 2. Add perceptual loss
    print('\n2. Train with perceptual loss...')
    λ_loss = {'Pixel': args.λ_pixel, 'Perceptual': args.λ_perct, 'SSIM': args.λ_ssim }
    print(λ_loss)

    print('Loading best weights from stage 1...')
    model.load_state_dict(torch.load(f'{CHECKPOINTS_FOLDER}{checkpoint}_stage_1_best.pth'))

    optimizer, scheduler = get_scheduler(model, args, lr=args.lr_2)
    train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args.n_epochs_2, checkpoint+'_stage_2', losses, λ_loss)


if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()