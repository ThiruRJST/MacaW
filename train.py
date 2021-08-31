import torch.nn as nn
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from dataset import AudioData
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from albumentations import Compose,Resize,Normalize
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from model import CNN_14,custom_effnet,custom_resnet
from torch.cuda.amp import autocast,GradScaler
from sklearn.metrics import label_ranking_average_precision_score


scaler = GradScaler()
loss_tr = nn.BCEWithLogitsLoss()
loss_val = nn.BCEWithLogitsLoss()
label_cols = ["species_"+str(m) for m in range(24)]

def seed_everything(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True



def train_loop(epoch,dataloader,model,optimizer,loss_fn,accum_iter=1,scheduler=None,device="cuda:0"):
    running_loss = 0.0
    imbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step,(img,label) in imbar:
        img = img.to(device).float()
        label = label.to(device).float()

        #Forward Pass
        with autocast():
            output = model(img)
            loss = loss_fn(output,label)
            if accum_iter>1:
                loss = loss/accum_iter
            
        scaler.scale(loss).backward()
        running_loss += loss.item()
            
        if((step+1) % accum_iter == 0) or((step+1) == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
    description = f"Epoch: {epoch} Loss: {running_loss/len(dataloader):.4f}"
    print(description)
    
    if scheduler is not None:
        scheduler.step()


def val_loop(epoch,dataloader,model,loss_fn):
    model.eval()
    val_loss = 0.0
    val_lwlrap = 0.0
    imbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step,(img,targets) in imbar:
        img = img.to("cuda:0").float()
        targets = targets.to("cuda:0").float()
        with torch.no_grad():
            outs = model(img)
            loss = loss_fn(outs,targets)
        val_loss += loss
        lwlrap = label_ranking_average_precision_score(y_score=outs.sigmoid().cpu(),y_true=targets.cpu())
        val_lwlrap += lwlrap

    print(f"Val_Loss: {val_loss/len(dataloader):.4f} Val_LWLRAP: {val_lwlrap/len(dataloader)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",help="Number of iterations to train the model")
    parser.add_argument("--data_path",help="Root Dir of the folder containing the data files")
    parser.add_argument("--csv_path",help="Train csv path")
    parser.add_argument("--fold",help="Fold number to train")
    parser.add_argument("--batch_size",help="Batch size to train with")
    parser.add_argument("--lr",help='Learning Rate')
    parser.add_argument("--device",help='CPU or CUDA')
    parser.add_argument("--Grad_Accum",help="Gradient Accumulation")


    args = parser.parse_args()
    num_epochs = int(args.epochs)
    data_path = args.data_path
    lr = args.lr
    device = args.device
    accum_iter = int(args.Grad_Accum)

    seed_everything()

    data = pd.read_csv(args.csv_path)
    df_train,df_valid = data[data.kfold != int(args.fold)].reset_index(drop=True),data[data.kfold == int(args.fold)].reset_index(drop=True)
    #print(df_valid)
    df_train_tar = df_train[label_cols].values
    df_valid_tar = df_valid[label_cols].values
    traindata = AudioData(root_dir=data_path,targets=df_train_tar,records=df_train.recording_id.values,transforms=Compose([Resize(200,600),Normalize(mean=0.485,std=0.229),ToTensorV2()]))
    valdata = AudioData(root_dir=data_path,targets=df_valid_tar,records=df_valid.recording_id.values,transforms=Compose([Resize(200,600),Normalize(mean=0.485,std=0.229),ToTensorV2()]))
    trainloader = DataLoader(traindata,batch_size=int(args.batch_size),num_workers=4)
    valloader = DataLoader(valdata,batch_size=int(args.batch_size),num_workers=4)
    model = custom_resnet().to(device)
    opt = optim.Adam(model.parameters(),lr=float(lr),weight_decay=1e-6)

    print("***** Model Training Started *****")
    for i in range(num_epochs):
        print(f"Epochs: {i}/{num_epochs}")
        train_loop(i,trainloader,model,opt,loss_tr,accum_iter,device=device)
        val_loop(i,valloader,model,loss_val)

