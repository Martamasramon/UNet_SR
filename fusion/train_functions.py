import numpy as np
import torch
from datetime       import datetime

CHECKPOINTS_ADC = '/cluster/project7/ProsRegNet_CellCount/UNet_SR/checkpoints/'
CHECKPOINTS_T2W = '/cluster/project7/ProsRegNet_CellCount/UNet/checkpoints/'

def get_checkpoint_name():
    now = datetime.now()
    checkpoint_file = f'{now.strftime("%d%m")}_{now.strftime("%H%M")}'
    return checkpoint_file

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def create_loss(losses):
    total_loss = 0.0
    total_losses = {}
    for l in losses:
        total_losses[l] = 0.0
    return total_loss, total_losses

def compute_loss(output, labels, losses, λ_loss):
    loss      = 0
    this_loss = {}
    for l in losses:
        if l!='Contrastive':
            this_loss[l] = losses[l](output, labels)
            loss += this_loss[l] * λ_loss[l]    
    return loss, this_loss

def update_loss(total_loss, loss, total_losses, this_loss):
    total_loss  += loss.item() 
    for l in this_loss:
        total_losses[l] += this_loss[l].item()
    return total_loss, total_losses

def average_loss(total_loss, total_losses, len_dataloader):
    total_loss /= len_dataloader
    for l in total_losses:
        total_losses[l] /= len_dataloader
    return total_loss, total_losses

def train(model_adc, model_t2w, optim_adc, optim_t2w, dataloader, losses, λ_loss):
    model_adc.train()
    model_t2w.train()
    total_loss_adc, total_losses_adc = create_loss(losses)
    total_loss_t2w, total_losses_t2w = create_loss(losses)
    total_contrast = 0
    
    for data in dataloader:
        # Get images
        adc_lowres, adc_highres, t2w = data['lowres'].float().cuda(), data['highres'].float().cuda(), data['T2W'].float().cuda()

        adc_pred, embed_adc = model_adc(adc_lowres)
        t2w_pred, embed_t2w = model_t2w(t2w)
                
        # Calculate loss
        loss_adc, this_loss_adc = compute_loss(adc_pred, adc_highres, losses, λ_loss)      
        loss_t2w, this_loss_t2w = compute_loss(t2w_pred, t2w,         losses, λ_loss)   
        
        contrast   = losses['Contrastive'](embed_adc, embed_t2w)
        total_loss = loss_adc + loss_t2w + λ_loss['Contrastive'] * contrast
        
        # Update optimizers
        optim_adc.zero_grad()
        optim_t2w.zero_grad()
        total_loss.backward()   
        optim_adc.step()
        optim_t2w.step()
        
        # Store losses
        total_loss_adc, total_losses_adc = update_loss(total_loss_adc, loss_adc, total_losses_adc, this_loss_adc)
        total_loss_t2w, total_losses_t2w = update_loss(total_loss_t2w, loss_t2w, total_losses_t2w, this_loss_t2w)
        total_contrast += contrast.item() 

    total_loss_adc, total_losses_adc = average_loss(total_loss_adc, total_losses_adc, len(dataloader))
    total_loss_t2w, total_losses_t2w = average_loss(total_loss_t2w, total_losses_t2w, len(dataloader))
    total_contrast /= len(dataloader)
    
    return total_loss_adc, total_losses_adc, total_loss_t2w, total_losses_t2w, total_contrast

def train_adc(model_adc, model_t2w, optim_adc, dataloader, losses, λ_loss):
    model_adc.train()
    total_loss_adc, total_losses_adc = create_loss(losses)
    total_contrast = 0
    
    for data in dataloader:
        # Get images
        adc_lowres, adc_highres, t2w = data['lowres'].float().cuda(), data['highres'].float().cuda(), data['T2W'].float().cuda()

        t2w_pred, embed_t2w = model_t2w(t2w)
        adc_pred, embed_adc = model_adc(adc_lowres, embed_t2w)
                
        # Calculate loss
        loss_adc, this_loss_adc = compute_loss(adc_pred, adc_highres, losses, λ_loss)              
        contrast   = losses['Contrastive'](embed_adc, embed_t2w)
        total_loss = loss_adc + λ_loss['Contrastive'] * contrast
        
        # Update optimizers
        optim_adc.zero_grad()
        total_loss.backward()   
        optim_adc.step()
        
        # Store losses
        total_loss_adc, total_losses_adc = update_loss(total_loss_adc, loss_adc, total_losses_adc, this_loss_adc)
        total_contrast += contrast.item() 

    total_loss_adc, total_losses_adc = average_loss(total_loss_adc, total_losses_adc, len(dataloader))
    total_contrast /= len(dataloader)
    
    return total_loss_adc, total_losses_adc, total_contrast


def evaluate(model_adc, model_t2w, dataloader, losses, λ_loss):
    model_adc.eval()
    model_t2w.eval()
    total_loss_adc, total_losses_adc = create_loss(losses)
    total_loss_t2w, total_losses_t2w = create_loss(losses)
    total_contrast = 0
    
    with torch.no_grad():
        for data in dataloader:
            # Get images
            adc_lowres, adc_highres, t2w = data['lowres'].float().cuda(), data['highres'].float().cuda(), data['T2W'].float().cuda()

            adc_pred, embed_adc = model_adc(adc_lowres)
            t2w_pred, embed_t2w = model_t2w(t2w)
            
            # Calculate loss
            loss_adc, this_loss_adc = compute_loss(adc_pred, adc_highres, losses, λ_loss)      
            loss_t2w, this_loss_t2w = compute_loss(t2w_pred, t2w,         losses, λ_loss)   
            
            contrast   = losses['Contrastive'](embed_adc, embed_t2w)
            total_loss = loss_adc + loss_t2w + λ_loss['Contrastive'] * contrast     

            # Store losses
            total_loss_adc, total_losses_adc = update_loss(total_loss_adc, loss_adc, total_losses_adc, this_loss_adc)
            total_loss_t2w, total_losses_t2w = update_loss(total_loss_t2w, loss_t2w, total_losses_t2w, this_loss_t2w)  
            total_contrast += contrast.item() 
            
    total_loss_adc, total_losses_adc = average_loss(total_loss_adc, total_losses_adc, len(dataloader))
    total_loss_t2w, total_losses_t2w = average_loss(total_loss_t2w, total_losses_t2w, len(dataloader))
    total_contrast /= len(dataloader)
    
    return total_loss_adc, total_losses_adc, total_loss_t2w, total_losses_t2w,  total_contrast

def evaluate_adc(model_adc, model_t2w, dataloader, losses, λ_loss):
    model_adc.eval()
    total_loss_adc, total_losses_adc = create_loss(losses)
    total_contrast = 0
    
    with torch.no_grad():
        for data in dataloader:
            # Get images
            adc_lowres, adc_highres, t2w = data['lowres'].float().cuda(), data['highres'].float().cuda(), data['T2W'].float().cuda()

            t2w_pred, embed_t2w = model_t2w(t2w)
            adc_pred, embed_adc = model_adc(adc_lowres, embed_t2w)
            
            # Calculate loss
            loss_adc, this_loss_adc = compute_loss(adc_pred, adc_highres, losses, λ_loss)                  
            contrast   = losses['Contrastive'](embed_adc, embed_t2w)
            total_loss = loss_adc + loss_t2w + λ_loss['Contrastive'] * contrast     

            # Store losses
            total_loss_adc, total_losses_adc = update_loss(total_loss_adc, loss_adc, total_losses_adc, this_loss_adc)
            total_contrast += contrast.item() 
            
    total_loss_adc, total_losses_adc = average_loss(total_loss_adc, total_losses_adc, len(dataloader))
    total_contrast /= len(dataloader)
    
    return total_loss_adc, total_losses_adc, total_contrast


def print_output(total, all_loss, name):
    output = f"{name} Loss: {total:.4f}. "
    
    if all_loss is not None:
        for loss in all_loss:
            output += f"{loss}: {all_loss[loss]:.4f}, "
    print(output)
    
    
def update_model(val_total, best_loss, model, name):
    if val_total < best_loss:
        best_loss = val_total
        torch.save(model.state_dict(), name + '_best.pth')
        print('Best model saved at', name + '_best.pth')

    torch.save(model.state_dict(), name + '_current.pth')
    return best_loss
    
    
def train_evaluate(model_adc, model_t2w, train_dl, test_dl, optim_adc, optim_t2w, sched_adc, sched_t2w, n_epochs, name_adc, name_t2w, losses, λ_loss):
    best_loss_adc = np.inf
    best_loss_t2w = np.inf
    
    for epoch in range(n_epochs):
        print('\nEpoch ', epoch)
        print('Learning rate ADC:', get_lr(optim_adc))
        print('Learning rate T2W:', get_lr(optim_t2w))
        
        ###### Training ######
        train_total_adc, train_all_adc, train_total_t2w, train_all_t2w, train_contrast = train(model_adc, model_t2w, optim_adc, optim_t2w, train_dl, losses, λ_loss)
        print_output(train_total_adc, train_all_adc, 'ADC Super-Resolution Training')
        print_output(train_total_t2w, train_all_t2w, 'T2W Reconstruction Training')
        print_output(train_contrast, None, 'Contrastive Training')

        ###### Evaluation ######
        val_total_adc, val_all_adc, val_total_t2w, val_all_t2w, val_contrast = evaluate(model_adc, model_t2w, test_dl, losses, λ_loss)
        print_output(val_total_adc, val_all_adc, 'ADC Super-Resolution Validation')
        print_output(val_total_t2w, val_all_t2w, 'T2W Reconstruction Validation')
        print_output(val_contrast, None, 'Contrastive Training')
        
        best_loss_adc = update_model(val_total_adc+val_contrast, best_loss_adc, model_adc, CHECKPOINTS_ADC + name_adc)
        best_loss_t2w = update_model(val_total_t2w+val_contrast, best_loss_t2w, model_t2w, CHECKPOINTS_T2W + name_t2w)

        sched_adc.step(val_total_adc+val_contrast)
        sched_t2w.step(val_total_t2w+val_contrast)
        
def train_evaluate_adc(model_adc, model_t2w, train_dl, test_dl, optim_adc, sched_adc, n_epochs, name_adc, losses, λ_loss):
    best_loss_adc = np.inf
    
    for epoch in range(n_epochs):
        print('\nEpoch ', epoch)
        print('Learning rate ADC:', get_lr(optim_adc))
        
        ###### Training ######
        train_total_adc, train_all_adc, train_contrast = train_adc(model_adc, model_t2w, optim_adc, train_dl, losses, λ_loss)
        print_output(train_total_adc, train_all_adc, 'ADC Super-Resolution Training')
        print_output(train_contrast, None, 'Contrastive Training')

        ###### Evaluation ######
        val_total_adc, val_all_adc, val_contrast = evaluate_adc(model_adc, model_t2w, test_dl, losses, λ_loss)
        print_output(val_total_adc, val_all_adc, 'ADC Super-Resolution Validation')
        print_output(val_contrast, None, 'Contrastive Training')
        
        best_loss_adc = update_model(val_total_adc+val_contrast, best_loss_adc, model_adc, CHECKPOINTS_ADC + name_adc)

        sched_adc.step(val_total_adc+val_contrast)
