import os
import argparse

import torchvision.io
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import PIL
from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO

def predict(image_location = './static/STR-TCGA-AAMALCER.jpg'):
    flag = "pathmnist"
    index = 99
    auc = 0.99894



    info = INFO[flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    device = 'cpu'

    model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)

    dir_path = os.path.join('./output', '%s_checkpoints' % (flag))
    restore_model_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc))

    model.load_state_dict(torch.load(restore_model_path, map_location= 'cpu')['net'])
    model.eval()
    #img = torchvision.io.read_image(image_location)
    img = PIL.Image.open(image_location)
    #npz_file = np.load(os.path.join('./input', "{}.npz".format(flag)))
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Grayscale(num_output_channels=1),
             transforms.Resize([28, 28]),
             transforms.Normalize(mean=[.5], std=[.5])])

    #img = npz_file['test_images'][0]
    #img = Image.fromarray(np.uint8(img))
    #img.save("test.jpeg")
    img = test_transform(img)
    #print(img.shape)

    img = img[None, :, :, :]
    output = model(img.to(device))
    m = nn.Softmax(dim=1)
    output = m(output).to(device)
    type = torch.argmax(output)
    type = type.item()
    ans = info["label"][str(type)]

    return ans

if __name__ == '__main__':
    print(predict())
