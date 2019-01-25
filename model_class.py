import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import cv2
import os
from math import exp
from torchvision import models
def get_class(img,model_ft):
   # model_ft=torch.load('best_model_res.pkl')
    transforms4=transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    img=transforms4(img)
    img=torch.unsqueeze(img,0)
    data=Variable(img.cuda())
    outputs=model_ft(data)
    _,preds=torch.max(outputs.data,1)
    ot=outputs.cpu().data.numpy().reshape(4)
    pi=[exp(x) for x in ot]
    p=pi[preds[0]]/sum(pi)
    return preds[0],p
#print(model_ft)
if __name__=='__main__':
    rootdir='/home/leo/data/nactev/activity_carrying/val/activity_carrying1/'
    files=os.listdir(rootdir)
    model=models.resnet34(pretrained=False)
    num_in=model.fc.in_features
    model.fc=torch.nn.Linear(num_in,4)
    model_ft=torch.load('./models/model.ckpt')
    model_ft={k.replace('module.',''):v for k,v in model_ft.items()}
    print(model_ft.keys())
    print(model.state_dict().keys())
    model.load_state_dict(model_ft)
    model=model.cuda()
    transforms4=transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    i=0
    j=0
    for file in files:
        file =rootdir+file
        img=Image.open(file).convert('RGB')
        #print(type(img))
        #print(img.size)
        preds=get_class(img,model)
        j+=1
        if(preds==0):
            i+=1
        print(i,j)
    print(i,j)
