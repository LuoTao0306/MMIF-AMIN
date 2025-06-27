import glob
from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
from Model import MODEL as net

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device('cuda:0')
model = net(in_channel=2)
path =  25
model_path = r"\models\model_{}.pth".format(path)
use_gpu = torch.cuda.is_available()
img1_dataset = sorted(glob.glob(r"/datasets/test_datasets/SPECT_MRI/*"))

img_name = []
for i in range(len(img1_dataset)):
    img_name.append(img1_dataset[i].split("\\")[-1])
    print(img1_dataset[i].split("\\")[-1])
print(img_name)
if use_gpu:
    model = model.cuda()
    model.cuda()
    model.load_state_dict(torch.load(model_path), strict=True)
else:
    state_dict = torch.load(model_path, map_location='cpu',strict=True)
    model.load_state_dict(state_dict)


def fusion():
    for num in range(len(img_name)):
        tic = time.time()
        path1 = r'/datasets/test_datasets/SPECT_MRI/SPECT/{}'.format(img_name[num])
        path2 = r'/datasets/test_datasets/SPECT_MRI/MRI/{}'.format(img_name[num])
        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')
        img1_org = img1
        img2_org = img2
        tran = transforms.ToTensor()
        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img
        model.eval()
        print(input_img.shape)
        out = model(input_img)
        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        imageio.imwrite(
            r'/fusion_out/Y/SPECT_MRI/{}'.format(img_name[num]),result)
        toc = time.time()
        print('end  {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))

if __name__ == '__main__':
    fusion()
    print(path)

