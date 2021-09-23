# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 03:17:05 2021

@author: Abdul Qayyum
"""

"""
Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156
Task 1 - Segmentation - Docker dummy example showing 
the input and output folders for the submission
"""

import sys  # For reading command line arguments
from glob import glob  # For listing files
import cv2  # For handling image loading & processing
import os  # for path creation
import numpy as np

# INPUT_PATH='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\docker_submission\\FetReg-segmentation-docker-example\\images\\input'
# OUTPUT_PATH='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\docker_submission\\FetReg-segmentation-docker-example\\output'

INPUT_PATH='/input'
OUTPUT_PATH='/output'

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
torch.cuda.device_count()  # print 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
# torch.cuda.device_count()  # still print 1
import torch.nn as nn
from builtins import range, zip
import random
from torch.utils.data import Dataset
from copy import deepcopy
from scipy.ndimage import map_coordinates, fourier_gaussian
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from skimage.transform import resize
from scipy.ndimage.measurements import label as lb
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class up_in(nn.Sequential):
    def __init__(self, num_input_features1, num_input_features2, num_output_features):
        super(up_in, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv1_1', nn.Conv2d(num_input_features1, num_input_features2,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('conv3_3', nn.Conv2d(num_input_features2, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        x = self.conv1_1(x)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class upblock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class up_out(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(up_out, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout = nn.Dropout2d(p=0.3)
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, y):
        y = self.up(y)
        y = self.conv3_3(y)
        y = self.dropout(y)
        y = self.norm(y)
        y = self.relu(y)
        return y


class DenseUNet(nn.Module):
    

    def __init__(self, num_channels, num_classes,growth_rate=48, block_config=(6, 12, 36, 24),
                num_init_features=96, bn_size=4, drop_rate=0,):

        super(DenseUNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.up1 = up_in(48*44, 48*46, 48*16)
        self.up2 = upblock(48*16, 48*8)
        self.up3 = upblock(48*8, 96)
        self.up4 = upblock(96,96)
        self.up5 = up_out(96,64)
        self.outconv = outconv(64,num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features.conv0(x)
        x0 = self.features.norm0(features)
        x0 = self.features.relu0(x0)
        x1 = self.features.pool0(x0)
        x1 = self.features.denseblock1(x1)
        x2 = self.features.transition1(x1)
        x2 = self.features.denseblock2(x2)
        x3 = self.features.transition2(x2)
        x3 = self.features.denseblock3(x3)
        x4 = self.features.transition3(x3)
        x4 = self.features.denseblock4(x4)
        
        y4 = self.up1(x3, x4)
        y3 = self.up2(x2, y4)
        y2 = self.up3(x1, y3)
        
        y1 = self.up4(x0, y2)
        y0 = self.up5(y1)
        out = self.outconv(y0)
        # out = F.softmax(out, dim=1)
        
        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
model1=DenseUNet(3, 4)
model2=DenseUNet(3, 4)
model3=DenseUNet(3, 4)
model4=DenseUNet(3, 4)
model5=DenseUNet(3, 4)


# modelUdense=ResUNet(n_channels, n_classes)
# print(modelUdense)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

path1="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\prediction\\trained_model\\WA\\modelDesnUnetAug2_fold1.pth"
model1.load_state_dict(torch.load('modelDesnUnetAug2_fold1.pth',map_location=torch.device('cpu')))
#model1.load_state_dict(torch.load('modelDesnUnetAug2_fold1.pth'))
model1.eval()

path2="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\prediction\\trained_model\\WA\\modelDensUnet_fold2_aug.pth"
model2.load_state_dict(torch.load('modelDensUnet_fold2_aug.pth',map_location=torch.device('cpu')))
#model2.load_state_dict(torch.load('modelDensUnet_fold2_aug.pth'))
model2.eval()

path3="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\prediction\\trained_model\\WA\\modelDensUnet_fold3_aug.pth"
model3.load_state_dict(torch.load('modelDensUnet_fold3_aug.pth',map_location=torch.device('cpu')))
#model3.load_state_dict(torch.load('modelDensUnet_fold3_aug.pth'))
model3.eval()

path4="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\prediction\\trained_model\\WA\\modelDensUnet_fold4_aug.pth"
model4.load_state_dict(torch.load('modelDensUnet_fold4_aug.pth',map_location=torch.device('cpu')))
#model4.load_state_dict(torch.load('modelDensUnet_fold4_aug.pth'))
model4.eval()

path5="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\prediction\\trained_model\\WA\\modelDensUnet_fold5_aug.pth"
model5.load_state_dict(torch.load('modelDensUnet_fold5_aug.pth',map_location=torch.device('cpu')))
#model5.load_state_dict(torch.load('modelDensUnet_fold5_aug.pth'))
model5.eval()
#model5.load_state_dict(torch.load(path5, map_location="cuda:0"))

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
        tpe = segmentation.dtype
        unique_labels = np.unique(segmentation)
        assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
        if order == 0:
            return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
        else:
            reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

            for i, c in enumerate(unique_labels):
                mask = segmentation == c
                reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
                reshaped[reshaped_multihot >= 0.5] = c
            return reshaped
        
input_file_list = glob(INPUT_PATH + "/*.png")
import ttach as tta
models=[model1,model2,model3,model4,model5]
for f in input_file_list:
    file_name = f.split("/")[-1]
    img = cv2.imread(f,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image=img
    image = resize(image, (256,256), order=1, mode="edge", clip=True, anti_aliasing=False)
    image=(np.asarray(image) / image.max()).astype('float32')
    image=torch.from_numpy(image).float().permute(2,0,1)
    img_t = image.unsqueeze(0)  
    preds = []
    for model_name in models:
        model = model_name
        tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        tta_model.eval()
        tta_model.to(device)
        with torch.no_grad():
            output = tta_model(img_t.to(device))
            preds.append(output)
                    
    preds = torch.stack(preds)

    predsm = preds.mean(0)
        
    output = torch.sigmoid(predsm)
    mask = torch.argmax(output[0,...], axis=0).float().cpu().numpy()
       
    mask_full=resize_segmentation(mask,(img.shape[0],img.shape[1]),order=1, cval=0)
    
    result = cv2.imwrite(OUTPUT_PATH + "/" + file_name, mask_full)
    if result==True:
        print(OUTPUT_PATH+'/' +file_name +' output mask saved')
    else:
        print('Error in saving file')
