import os
import argparse

from tqdm import tqdm
import pandas as pd
import joblib
import glob

from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Model import MODEL as net
from losses import ssim_spect, ssim_mri,RMI_spect,RMI_mri,MP_AG_loss,MP_EN_loss
device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="train", help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[0.2, 0.8,1,1], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)')
    args = parser.parse_args()
    return args

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, train_loader, model, criterion_ssim_spect, criterion_ssim_mri,criterion_RMI_spect,criterion_RMI_mri,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ssim_spect = AverageMeter()
    losses_ssim_mri = AverageMeter()
    losses_RMI_spect = AverageMeter()
    losses_RMI_mri= AverageMeter()
    weight = args.weight#The adaptive loss function is used by default; when manual mode is employed, settings need to be configured.
    model.train()
    for i, (input,spect,mri)  in tqdm(enumerate(train_loader), total=len(train_loader)):
        if use_gpu:
            input = input.cuda()
            spect=spect.cuda()
            mri=mri.cuda()
        else:
            input = input
            spect=spect
            mri=mri
        out = model(input)
        mri_ag,pet_ag = MP_AG_loss(mri,spect)
        mri_en,pet_en = MP_EN_loss(mri,spect)
        loss_ssim_spect= pet_ag * criterion_ssim_spect(out, spect)
        loss_ssim_mri= mri_ag * criterion_ssim_mri(out, mri)
        loss_RMI_spect=pet_en * criterion_RMI_spect(out,spect)
        loss_RMI_mri = mri_en * criterion_RMI_mri(out,mri)
        loss = loss_ssim_spect + loss_ssim_mri+loss_RMI_spect+ loss_RMI_mri
        losses.update(loss.item(), input.size(0))
        losses_ssim_spect.update(loss_ssim_spect.item(), input.size(0))
        losses_ssim_mri.update(loss_ssim_mri.item(), input.size(0))
        losses_RMI_spect.update(loss_RMI_spect.item(), input.size(0))
        losses_RMI_mri.update(loss_RMI_mri.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ssim_spect', losses_ssim_spect.avg),
        ('loss_ssim_mri', losses_ssim_mri.avg),
        ('loss_RMI_spect', losses_RMI_spect.avg),
        ('loss_RMI_mri', losses_RMI_mri.avg),
    ])
    return log

class my_GetDataset(Dataset):
    def __init__(self, spect_imageFolderDataset, mri_imageFolderDataset,transform=None):
        """
        :param spect_imageFolderDataset: MRI dataset
        :param mri_imageFolderDataset: SPECT dataset
        :param transform:
        """
        self.spect_imageFolderDataset = spect_imageFolderDataset#MRI
        self.mri_imageFolderDataset = mri_imageFolderDataset#SPECT
        self.transform = transform
    def __getitem__(self, index):
        spect = str(self.spect_imageFolderDataset[index])
        mri = str(self.mri_imageFolderDataset[index])
        spect = Image.open(spect).convert('L')
        mri = Image.open(mri).convert('L')
        if self.transform is not None:
            tran = transforms.ToTensor()
            spect=tran(spect)
            mri= tran(mri)
            input = torch.cat((spect, mri), -3)
            return input, spect,mri
    def __len__(self):
        return len(self.mri_imageFolderDataset)

def main():
    args = parse_args()
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True
    training_dir_spect = r"datasets/train_dataset/SPECT_MRI/SPECT/*"
    folder_dataset_train_spect =sorted(glob.glob(training_dir_spect))
    training_dir_mri = r"datasets/train_dataset/SPECT_MRI/MRI/*"
    folder_dataset_train_mri= sorted(glob.glob(training_dir_mri))
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])
    my_datase = my_GetDataset(folder_dataset_train_spect, folder_dataset_train_mri,transform=transform_train)
    train_loader = DataLoader(my_datase, shuffle=True, num_workers=4)
    model = net(in_channel=2)
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model
    criterion_ssim_spect = ssim_spect
    criterion_ssim_mri = ssim_mri
    criterion_RMI_spect = RMI_spect
    criterion_RMI_mri= RMI_mri
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)

    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'loss_ssim_spect',
                                'loss_ssim_mri',
                                'loss_RMI_spect',
                                'loss_RMI_mri',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        train_log = train(args, train_loader, model, criterion_ssim_spect, criterion_ssim_mri,criterion_RMI_spect,criterion_RMI_mri, optimizer, epoch)     # 训练集
        print('loss: %.4f - loss_ssim_spect: %.4f - loss_ssim_mri: %.4f - loss_RMI_spect: %.4f - loss_RMI_mri: %.4f '
              % (train_log['loss'],
                 train_log['loss_ssim_spect'],
                 train_log['loss_ssim_mri'],
                 train_log['loss_RMI_spect'],
                 train_log['loss_RMI_mri'],
                 ))


        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['loss_ssim_spect'],
            train_log['loss_ssim_mri'],
            train_log['loss_RMI_spect'],
            train_log['loss_RMI_mri'],
        ], index=['epoch', 'loss', 'loss_ssim_spect', 'loss_ssim_mri', 'loss_RMI_spect', 'loss_RMI_mri'])
        log = log._append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)
if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()

