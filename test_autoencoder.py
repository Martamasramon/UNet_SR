#### Train MSSSIM_VAE (recommended default for latent-diffusion SR)
from pythae.models          import AutoModel
from dataset.pythae_dataset import PythaeDataset
from dataset.dataset        import MyDataset  
from model.test_functions   import visualize_results, evaluate_results
from torch.utils.data       import DataLoader
from arguments              import args
import torch 

def get_prediction_from_model(model, x):
    """
    Works for both: your SR nets (return tensor or (tensor,_))
    and PyThae models (return ModelOutput with .reconstruction).
    """
    out = model(x)                    # could be tensor / tuple / ModelOutput
    # PyThae
    if hasattr(out, "reconstruction"):
        return out.reconstruction
    # tuple from your SR model(s)
    if isinstance(out, (tuple, list)):
        return out[0]
    # plain tensor
    return out
    
def main():
    # Set device
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    print('Creating dataset...')
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'

    val_base = MyDataset(
        data_folder, 
        data_type       = 'test', 
        img_size        = args.img_size, 
        down_factor     = args.down,
        is_finetune     = args.finetune, 
        use_mask        = args.masked
    )
    val_ds = PythaeDataset(val_base)

    # Peek one sample to infer shape
    C, H, W = val_ds[0]["data"].shape
    print("Sample shape:", (C, H, W))
    
    # Load models
    if args.autoencoder == 'MSSSIM':
        model_path = f'./autoencoder/msssim_vae/{args.checkpoint}'
    elif args.autoencoder == 'VAEGAN':
        model_path = f'./autoencoder/vaegan/{args.checkpoint}'
    
    print(f"Loading ADC-SR RUNet weights from {model_path}")
    model = AutoModel.load_from_folder(model_path) 
    model.eval()
    
    ## VISUALISE
    save_name = args.autoencoder +'_HistoMRI' if args.finetune else args.autoencoder+'_PICAI' 
    visualize_results(model, None, val_ds, device, save_name, args.use_T2W, args.test_bs, args.fusion, ae_mode=True)

    ## EVALUATE
    dataloader = DataLoader(val_ds,  batch_size=args.test_bs,  shuffle=False)
    evaluate_results(model, None, dataloader, device, args.test_bs, args.fusion, ae_mode=True)
    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()