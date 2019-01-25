from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from net import simpleconv3
from data import  MyDataSet
from data import  MyDataSet
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from opts import parser
args=parser.parse_args()
model_name=args.model_name
#print(dir)
import datetime
a=datetime.datetime.now()
b=datetime.datetime.strftime(a,'-%Y-%m-%d-%H-%M-%S')
n_model_name=model_name+b
writer = SummaryWriter('runs/'+model_name+'/'+n_model_name)

def train_model(model, criterion, optimizer, scheduler, num_epochs=52):
    global model_name
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                torch.save(model.state_dict(),'runs/'+model_name+'/'+model_name+str(epoch)+'.ckpt')
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    return model

if __name__ == '__main__':

    #global model_name
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
    }

    bs=32
    if model_name=='resnet34':
        modelclc=models.resnet34(pretrained=True)
    if model_name=='resnet152':
        bs=8
        modelclc=models.resnet152(pretrained=True)
    if model_name=='resnet18':
        modelclc=models.resnet18
    if model_name=="resnet50":
        modelclc=models.resnet50(pretrained=True)
    image_datasets = {x: MyDataSet(x,data_transforms[x]) for x in ['train', 'val']}
    weights={x:MyDataSet(x,data_transforms[x]).weights for x in['train','val']}
    print(weights['train'])
    sampler={x:WeightedRandomSampler(weights[x],num_samples=len(image_datasets[x]),replacement=True) for x in ['train','val']}

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=bs,
                                                 sampler=sampler[x],
                                                 num_workers=8) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()

    #modelclc = simpleconv3()
    num_ftrs=modelclc.fc.in_features
    modelclc.fc = nn.Linear(num_ftrs, 4)
    print(modelclc)
    if use_gpu:
        modelclc = torch.nn.DataParallel(modelclc,device_ids=[0,1]).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(modelclc.parameters(), lr=0.1, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    modelclc = train_model(model=modelclc,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=52)

    #os.mkdir('models')
    torch.save(modelclc.state_dict(),'runs/'+model_name+'/'+model_name+'.ckpt')
