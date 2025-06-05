import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.transforms import create_transforms
from dataset.dataset    import MyDataset
from runet.runet import RUNet

class RUNetVisualizer:
    def __init__(self, drop_first, drop_last, img_folder, is_pretrain, img_size=64, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        train_transforms, test_transforms = create_transforms(img_size)
        self.train_dataloader   = MyDataset(img_folder, train_transforms, is_pretrain=is_pretrain, is_train=True)
        self.test_dataloader    = MyDataset(img_folder, test_transforms,  is_pretrain=is_pretrain, is_train=False)
        self.device=device
        self.model = RUNet(drop_first, drop_last).to(device)
        self.model.eval()
        self.is_pretrain = is_pretrain

    def format_image(self, img):
        return np.squeeze((img).cpu().numpy())

    def visualize_runet(self, checkpoint, eval_test=True, batch_size=5,seed=1):
        dataloader = self.test_dataloader if eval_test else self.train_dataloader

        print(f"Loading RUNet weights from {checkpoint}")
        self.model.load_state_dict(torch.load(checkpoint))

        fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(3*3,3*batch_size))
        axes[0,0].set_title('Low res (linear interpolation)')
        axes[0,1].set_title('Super resolution')
        axes[0,2].set_title('High res')
        np.random.seed(seed)
        indices = np.random.choice(np.arange(len(dataloader)),batch_size,replace=False)

        with torch.no_grad():
            for i,ind in enumerate(indices):
                sample = dataloader[ind]
                img       = sample["image"].unsqueeze(0).float().to(self.device)
                label     = sample["label"].unsqueeze(0).float().to(self.device)
            
                img_super = self.model(img)

                im0 =axes[i,0].imshow(self.format_image(img),        cmap='gray',vmin=0,vmax=1)
                axes[i,0].axis('off')
                plt.colorbar(im0, ax=axes[i,0])
                im1 =axes[i,1].imshow(self.format_image(img_super),  cmap='gray',vmin=0,vmax=1)
                axes[i,1].axis('off')
                plt.colorbar(im1, ax=axes[i,1])
                im2 =axes[i,2].imshow(self.format_image(label),      cmap='gray',vmin=0,vmax=1)
                axes[i,2].axis('off')
                plt.colorbar(im2, ax=axes[i,2])
                
            fig.tight_layout(pad=0.25)
            img_suffix = 'PICAI' if self.is_pretrain else 'HistoMRI'
            plt.savefig(f'./results/image_{checkpoint[24:-4]}_{img_suffix}.jpg')
            plt.close()
            
    
            
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   type=str,  default='checkpoints_2605_1715_stage_1_best')
parser.add_argument('--img_size',     type=int,  default=64)
parser.add_argument('--drop_first',   type=float,default=0.1)
parser.add_argument('--drop_last',    type=float,default=0.5)
parser.add_argument('--batch_size',   type=int,  default=5)

parser.add_argument('--img_folder',   type=str,  default='/cluster/project7/backup_masramon/IQT/PICAI/ADC/')
parser.add_argument('--is_pretrain',  action='store_true')
parser.add_argument('--finetune',     dest='is_pretrain', action='store_false')
parser.set_defaults(is_pretrain=True)
args, unparsed = parser.parse_known_args()
print('\n',args)

## VISUALISE RESULTS
checkpoint_unet = f"checkpoints/{args.checkpoint}.pth"

visualizer_runet = RUNetVisualizer(args.drop_first, args.drop_last, args.img_folder, args.is_pretrain, img_size=args.img_size)
visualizer_runet.visualize_runet(checkpoint_unet, batch_size=args.batch_size)

# ## EVALUATE
# checkpoint_unet = "checkpoints/perceptual_loss_RUNET_var_blur.pth"
# evaluator_runet = RUNetEvaluation()
# evaluator_runet.evaluate(checkpoint_unet)
