from __future__ import print_function, division

import sys

print(sys.path)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train']}
dataloaders = dataloaders['train']

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes


# Training
# n_training_samples = 20000
# train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
#                                                sampler=train_sampler, num_workers=2)
# Validation
# n_val_samples = 5000
# val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

# Test
# n_test_samples = 5000
# test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


# DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory)
def get_train_loader(batch_size):
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                           sampler=train_sampler, num_workers=2)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train']}
    return dataloaders['train']


# Test and validation loaders have constant batch sizes, so we can define them directly
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
# val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)


def print_data(net, net_base, img_list, e_size, g_size, o_size):
    global device

    count_erika = 0
    count_gustavo = 0
    count_other = 0

    for i in range(len(img_list)):

        net_base.eval()
        output = net(net_base(img_list[i]))
        # print(output[0], ' - ', list_name[i])
        ind = torch.argmax(output)
        # print(ind.item())
        if ind.item() == 0:
            # print('Erika')
            if i < e_size:
                count_erika += 1
        
        if ind.item() == 1:
            # print('Gustavo')
            if i < g_size + e_size and i >= e_size:
                count_gustavo += 1

        if ind.item() == 2:
            # print('OUTRO')
            if i < o_size + g_size + e_size and i >= g_size + e_size:
                count_other += 1

        # print('CLASS:' + str(classes[i.item()]))

        # cv2.waitKey(0)

    print('Accuracy Erika: ', math.floor(10000 * count_erika / e_size) / 100, '%',
        'Accuracy Gustavo:', math.floor(10000 * count_gustavo / g_size) / 100, '% '
        'Accuracy other:', math.floor(10000 * count_other / o_size) / 100, '%')


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


def trainNet(net, net_base, batch_size, n_epochs, learning_rate, debug):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # -------------------
    # Setting up directories for training data set
    # -------------------

    list_train = []
    for i in range(1, 176):
        list_train.append('dataset/train/erika/IMG' + str(i) + '.jpg')

    erika_size_t = len(list_train)

    for i in range(1, 154):
        list_train.append('dataset/train/gustavo/IMG' + str(i) + '.jpg')

    gustavo_size_t = len(list_train) - erika_size_t

    for i in range(1, 201):
        list_train.append('dataset/train/unknown/IMG' + str(i) + '.jpg')

    other_size_t = len(list_train) - gustavo_size_t - erika_size_t

    # -------------------
    # Setting up directories for validation data set
    # -------------------

    list_name = []
    for i in range(1, 115):
        list_name.append('dataset/val/erika/IMG' + str(i) + '.jpg')

    erika_size = len(list_name)

    for i in range(1, 56):
        list_name.append('dataset/val/gustavo/IMG' + str(i) + '.jpg')    

    gustavo_size = len(list_name) - erika_size

    for i in range(1, 201):
        list_name.append('dataset/val/unknown/IMG' + str(i) + '.jpg')    

    other_size = len(list_name) - gustavo_size - erika_size

    # -------------------

    list_name = setup_data(list_name)
    list_train = setup_data(list_train)

    # Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 2
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data

            # Wrap them in a Variable object
            inputs, labels = inputs.to(device), labels.to(device)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net_base(inputs)
            outputs = net(outputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        if debug:
            print('Validation results: ')
            print_data(net, net_base, list_name, erika_size, gustavo_size, other_size)
            print('Training results: ')
            print_data(net, net_base, list_train, erika_size_t, gustavo_size_t, other_size_t)
            print('-' * 30, '\n')
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        # for inputs, labels in val_loader:
        #    # Wrap tensors in Variables
        #    inputs, labels = inputs.to(device), labels.to(device)
        #
        #    # Forward pass
        #    val_outputs = net(inputs)
        #   val_loss_size = loss(val_outputs, labels)
        #    total_val_loss += val_loss_size.data

        # print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    
    print('Validation results: ')
    print_data(net, net_base, list_name, erika_size, gustavo_size, other_size)
    print('Training results: ')
    print_data(net, net_base, list_train, erika_size_t, gustavo_size_t, other_size_t)
    print('-' * 30, '\n')

resnet = InceptionResnetV1(classify=False, pretrained='vggface2', num_classes=(3))
model = resnet.to(device)
output_net = OutputLayer().to(device)
trainNet(output_net, model, batch_size=16, n_epochs=5, learning_rate=0.001, debug=False)

print('Saving output model...')
torch.save(output_net, './model_output_facerecognition_5e16b')

print('Saving base model...')
torch.save(model, './model_base_facerecognition_5e16b')