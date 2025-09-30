#### Train MSSSIM_VAE (recommended default for latent-diffusion SR)
from pythae.pipelines import TrainingPipeline
from pythae.trainers  import BaseTrainerConfig as BaseConfig
from pythae.trainers  import CoupledOptimizerAdversarialTrainerConfig as AdverseConfig
from pythae.models    import MSSSIM_VAE, MSSSIM_VAEConfig, VAEGAN, VAEGANConfig

from dataset.pythae_dataset  import PythaeDataset
from dataset.dataset import MyDataset  
from arguments       import args
from torch.optim     import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'

    train_base = MyDataset(
        data_folder, 
        data_type       = 'train', 
        img_size        = args.img_size, 
        down_factor     = args.down,
        is_finetune     = args.finetune, 
        use_mask        = args.masked
    )
    test_base = MyDataset(
        data_folder, 
        data_type       = 'test', 
        img_size        = args.img_size, 
        down_factor     = args.down,
        is_finetune     = args.finetune, 
        use_mask        = args.masked
    )
    train_ds = PythaeDataset(train_base)
    test_ds  = PythaeDataset(test_base)

    # Peek one sample to infer shape
    x0 = train_ds[0]["data"]
    C, H, W = x0.shape
    print("Sample shape:", (C, H, W))

    trainer_default = {
        'num_epochs'                  : args.n_epochs,
        'learning_rate'               : args.lr,
        'optimizer_cls'               : AdamW,                 
        'optimizer_params'            : {"weight_decay": 1e-5, "betas": (0.9, 0.999)},
        'scheduler_cls'               : CosineAnnealingLR,     # smooth decay over training
        'scheduler_params'            : {"T_max": args.n_epochs, "eta_min": 1e-6},

        'per_device_train_batch_size' : args.train_bs,
        'per_device_eval_batch_size'  : args.test_bs,
        'gradient_accumulation_steps' : 1,
        
        'train_dataloader_num_workers': 4,
        'eval_dataloader_num_workers' : 4,
        'amp'                         : True,         # mixed precision for speed/VRAM 
        'steps_saving'                : 1,            # save each epoch 
        'keep_best_on_train'          : False,        
        'seed'                        : 123
    }
    
    if args.autoencoder == 'MSSSIM':
        # Set trainer config (you can add schedulers/WD if you like)
        trainer_cfg = BaseConfig(
            output_dir="./autoencoder/msssim_vae",
            **dict(trainer_default)
        ) 

        model_cfg = MSSSIM_VAEConfig(
            input_dim           = (C, H, W),
            latent_dim          = 8,        # Try 8 // 12 // 16
            reconstruction_loss = "mse"     # MSSSIM is added internally; keep mse here
        )
        
        model = MSSSIM_VAE(model_cfg)

    elif args.autoencoder == 'VAEGAN':
        # Adversarial trainer config (different class!)
        trainer_cfg = AdverseConfig(
            output_dir="autoencoder/vaegan",
            **dict(trainer_default)
        )

        model_cfg = VAEGANConfig(
            input_dim               = (C, H, W),
            latent_dim              = 8,
            reconstruction_loss     = "mse",        # 'mse' or 'bce'
            adversarial_loss_scale  = 0.2,          # keep small for MRI fidelity
            uses_default_encoder    = True,         # provide custom conv nets if your images are large
            uses_default_decoder    = True,
            uses_default_discriminator = True
        )

        model = VAEGAN(model_cfg)

    pipeline = TrainingPipeline(training_config=trainer_cfg, model=model)
    pipeline(train_data=train_ds, eval_data=test_ds)

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()