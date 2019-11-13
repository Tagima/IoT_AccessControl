from __future__ import print_function, division

import sys

#print(sys.path)
to_remove = []
for i in range(len(sys.path)):
    if sys.path[i].find('python2.7') > 0:
        sys.path.append(sys.path[i])
        to_remove.append(sys.path[i])

for i in range(len(to_remove)):
    sys.path.remove(to_remove[i])

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from facenet_pytorch import InceptionResnetV1
from cnn import *

import cv2
import time
import os
import math
import numpy as np

threshold = 0.4     # Threshold to determine the releability on the image.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Test and validation loaders have constant batch sizes, so we can define them directly
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
# val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)


def print_data(net, net_base, img_list, t_size):
    global device

    count_erika = 0
    count_gustavo = 0
    count_other = 0

    for i in range(len(img_list)):

        net_base.eval()
        net.eval()
        output = net(net_base(img_list[i]))

        output_vec = output.tolist()[0]
        ind = max(output_vec)

        if ind < threshold:     # Can't rely on the sample, just set it as unknown
            face_class = 2
        else:
            face_class = output_vec.index(ind)

        if face_class == 0:
            count_erika += 1
        
        if face_class == 1:
            count_gustavo += 1
            
        if face_class == 2:
            count_other += 1

    if count_erika >= 3:    # Existe boa confiança de que a pessoa que apareceu sou eu
        print("Erika")

    elif count_gustavo >= 3:
        print("Gustavo")

    else:
        print("Unknown")

def setup_data(list_in):
    global device

    img_list = []

    for i in range(len(list_in)):
        a = cv2.imread(list_in[i])
        r = cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA)

        trans = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = trans(r)

        image = image.unsqueeze(0)
        image = image.to(device=device, dtype=torch.float)
        img_list.append(image)

    return img_list
  

output_net = torch.load('./model_output_facerecognition_5e16b')
model_out = output_net.to(device)

resnet = torch.load('./model_base_facerecognition_5e16b')
#resnet = InceptionResnetV1(classify=False, pretrained='vggface2', num_classes=(3))
model = resnet.to(device)

# -------------------
# Setting up directories for test data set
# -------------------

list_train = []
for i in range(1, 6):
    list_train.append('/var/tmp_app/sample_' + str(i) + '.jpg')
    #list_train.append('test/IMG' + str(i) + '.jpg')

test_size_t = len(list_train)

# -------------------

list_train = setup_data(list_train)

print('\n')
print('Result: ')
print_data(model_out, model, list_train, test_size_t)
print('\n')
