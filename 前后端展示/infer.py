import os
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import zipfile
from loss import get_loss
from model import get_model
from data import get_data

index = 0

@torch.no_grad()
def infer(cp_path, model_name,  threshold, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = os.listdir(cp_path)
    models = []
    
    for fold in folds:
        model_checkpoints = os.listdir(os.path.join(cp_path, fold))
        for model_checkpoint in model_checkpoints:
            model = get_model(model_name).to(device)
            # print(os.path.join(path, fold, model_checkpoint))
            weight = torch.load(
                os.path.join(cp_path, fold, model_checkpoint), map_location=device
            )
            model.load_state_dict(weight)
            model.eval()
            models.append(model)
    inputs = get_data(image_path)
    inputs0 = inputs.reshape(1, 3, 320, 640).to(device)
    inputs1 = inputs0.flip(dims=[2]).to(device)
    inputs2 = inputs0.flip(dims=[3]).to(device)
    inputs3 = inputs0.flip(dims=[2, 3]).to(device)
    out = 0
    for model in models:
        out0 = model(inputs0)
        out1 = model(inputs1).flip(dims=[2])
        out2 = model(inputs2).flip(dims=[3])
        out3 = model(inputs3).flip(dims=[2, 3])
        out = out + out0 + out1 + out2 + out3
    out = out / len(models)
    threshold = threshold
    out = torch.where(
        out >= threshold, torch.tensor(255, dtype=torch.float).to(device), out
    )
    out = torch.where(
        out < threshold, torch.tensor(0, dtype=torch.float).to(device), out
    )
    out = out.detach().cpu().numpy().reshape(1, 320, 640)
    img = Image.fromarray(out[0].astype(np.uint8))
    img = img.convert("1")
   
    
    return img
       

        

def detect(image_path):
    
    return infer(
    model_name="deeplabv3p",
    cp_path="D:\暑期考核\cp_v1",
    threshold=0.5,
    image_path=image_path
    )



