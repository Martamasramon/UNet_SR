import sys
sys.path.append("../UNet/")
import torch
from runet_t2w.runetv2 import RUNet

BEST_CHECKPOINT = '/cluster/project7/ProsRegNet_CellCount/UNet/checkpoints/checkpoints_0306_1947_stage_1_best.pth'
model_setup     = {
    'drop_first': 0.1,
    'drop_last': 0.5,
}

def get_t2w_model():
    model = RUNet(model_setup['drop_first'], model_setup['drop_last']).cuda()
    model.load_state_dict(torch.load(BEST_CHECKPOINT))
    model.eval()
    return model

def get_embedding(model, img):
    with torch.no_grad():
        return model.get_embedding(img)

import numpy as np

model = get_t2w_model()
test  = torch.zeros((128,128)).unsqueeze(0).unsqueeze(0).float().cuda()   
embedding = get_embedding(model, test)
print(embedding.shape)