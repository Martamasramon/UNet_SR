import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size',       type=int,  default=64)
parser.add_argument('--train_bs',       type=int,  default=16)
parser.add_argument('--test_bs',        type=int,  default=1)  # 15 for test
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)

parser.add_argument('--use_T2W',    action='store_true')
parser.add_argument('--use_histo',  action='store_true')
parser.set_defaults(use_T2W=False)
parser.set_defaults(use_histo=False)

# For a single training phase
parser.add_argument('--n_epochs',   type=int,   default=50)
parser.add_argument('--lr',         type=float, default=1e-6)

# Multiple training phases
parser.add_argument('--n_epochs_1', type=int,   default=50)
parser.add_argument('--n_epochs_2', type=int,   default=50)
parser.add_argument('--lr_1',       type=float, default=1e-7)
parser.add_argument('--lr_2',       type=float, default=1e-7)

parser.add_argument('--factor',     type=float, default=0.5)
parser.add_argument('--patience',   type=int,   default=4)
parser.add_argument('--cooldown',   type=int,   default=2)
parser.add_argument('--lr_factor',  type=float, default=0.1)

parser.add_argument('--λ_pixel',    type=float, default=10.0)
parser.add_argument('--λ_perct',    type=float, default=0.01)
parser.add_argument('--λ_ssim',     type=float, default=1.0) # 0 for base training
parser.add_argument('--λ_contrast', type=float, default=1.0)

# Checkpoint
parser.add_argument('--checkpoint_adc', type=str,   default='pretrain_PICAI')   # 'pretrain_PICAI_cont' for fusion
parser.add_argument('--checkpoint_t2w', type=str,   default='default_64')       # 'default_64_cont' for fusion
parser.add_argument('--checkpoint',     type=str,   default=None) # 'pretrain_PICAI' for test
parser.add_argument('--save_as',        type=str,   default=None)

parser.add_argument('--finetune',       action='store_true')
parser.add_argument('--fusion',         action='store_true')
parser.add_argument('--surgical_only',  action='store_true')
parser.set_defaults(finetune=False)
parser.set_defaults(fusion=False)
parser.set_defaults(surgical_only=False)

args, unparsed = parser.parse_known_args()