import argparse
import os
import math
from functools import partial

import yaml
import torch
from torchvision import transforms
from PIL import Image

import numpy as np
import datasets
import models
import utils
from utils import make_coord
from torchvision.utils import save_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/xqgao/2025/MIT/code/GS/ContinuousSR-main/butterflyx4.png', help='Input image file')
    parser.add_argument('--model', type=str, default='/home/xqgao/2025/MIT/code/GS/ContinuousSR-main/ContinuousSR.pth', help='Path to the model file')
    parser.add_argument('--scale', type=str, default='4,4', help='Scaling factors for the image (default: 4,4)')
    parser.add_argument('--output', type=str, default='/home/xqgao/2025/MIT/code/GS/ContinuousSR-main/output.png', help='Output image file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU index to use (default: 0)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB')).cuda()
    
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    s1, s2 = list(map(int, args.scale.split(',')))
    scale = torch.tensor([[s1, s2]]).cuda()
  
    with torch.no_grad():
        pred = model(img.unsqueeze(0), scale).squeeze(0)
        pred = pred.clamp(0,1)
        
    transforms.ToPILImage()(pred).save(args.output)
    print("finished!")
