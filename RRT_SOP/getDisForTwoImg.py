import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.ingredient import get_model_EX
import time
device = torch.device('cuda:0' if torch.cuda.is_available() and not False else 'cpu')
Matcher = None


def main(resume):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not False else 'cpu')
    num_classes = 11318
    # 获取resnet模型
    model = get_model_EX(num_classes)
    state_dict = torch.load(resume, map_location=torch.device('cpu'))
    if 'state' in state_dict:
        state_dict = state_dict['state']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    # if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    global Matcher
    Matcher = model

def getDis(Img1Path,Img2Path):
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor()])
    image1 = data_transform(Image.open(Img1Path))
    image2 = data_transform(Image.open(Img2Path))
    features1= Matcher(image1)[2]
    features2= Matcher(image2)[2]
    current_scores, rrtfeatures, _, _ = Matcher(None, True,
                                                src_global=None, src_local=features1.to(device),
                                                tgt_global=None, tgt_local=features2.to(device))
    return current_scores

# getDis("1.JPG", "2.JPG")
main("rrt_sop_ckpts/rrt_r50_sop_rerank_finetune.pt")