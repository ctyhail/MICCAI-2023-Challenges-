import os
import torch
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import get_model
from data import get_data





def zip_files(file_paths, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            zipf.write(file)


## 普通推理
@torch.no_grad()
def infer(model_name, checkpoint_path, img_save_path, zip_out_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    testdata = get_data("test")
    for i, inputs in tqdm(enumerate(testdata)):
        inputs = inputs.reshape(1, 3, 320, 640).to(device)
        out = model(inputs)
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
        img.save(img_save_path + testdata.name[i])

    # 打包图片
    file_paths = [
        img_save_path + i for i in os.listdir(img_save_path) if i[-3:] == "png"
    ]
    zip_files(file_paths, zip_out_path)


# 带tta的推理
@torch.no_grad()
def infer_with_tta(
    model_name, checkpoint_path, img_save_path, zip_out_path, threshold=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    testdata = get_data("test")
    for i, inputs in tqdm(enumerate(testdata)):
        inputs = inputs.reshape(1, 3, 320, 640).to(device)
        inputs1 = inputs.flip(dims=[2]).to(device)#在指定维度进行翻转
        inputs2 = inputs.flip(dims=[3]).to(device)
        inputs3 = inputs.flip(dims=[2, 3]).to(device)
        out = model(inputs)
        out1 = model(inputs1).flip(dims=[2])
        out2 = model(inputs2).flip(dims=[3])
        out3 = model(inputs3).flip(dims=[2, 3])
        out = (out + out1 + out2 + out3) / 4
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
        img.save(img_save_path + testdata.name[i])

    file_paths = [
        img_save_path + i for i in os.listdir(img_save_path) if i[-3:] == "png"
    ]
    zip_files(file_paths, zip_out_path)



infer_with_tta(
    model_name="deeplabv3p",
    checkpoint_path="",
    img_save_path="infers/",
    zip_out_path="infers.zip",
    threshold=0.5,
)

