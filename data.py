import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
class MyDataSet(Dataset):
    def __init__(self,tv, transform=None, target_transform=None):
        imgs=[]
        self.tv=tv
        self.cls_num=[0,0,0,0]
        self.weights=[]
        txt='/home/leo/data/nactev/list/img_2018actev_'+self.tv+'_file.txt'
        f=open(txt,'r')
        lines=f.readlines()
        for line in lines:
            line=line.strip().split()
            imgs.append((line[0],int(line[1])))
            self.cls_num[int(line[1])]+=1
        for line in lines:
            line=line.strip().split()
            self.weights.append( 1.0/ self.cls_num [int(line[1])])
        f.close()
        self.imgs=imgs
        self.transform=transform
        self.target_transform=target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img=Image.open(fn).convert('RGB')
        if self.transform!=None:
            img=self.transform(img)
        label=np.array(label)
        label=torch.from_numpy(label).long()
        return img,label
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
'''my_trans=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       # ])
train_data=MyDataSet('train',my_trans)
val_data=MyDataSet('val',my_trans)
l1=len(train_data)
l2=len(val_data)
print(l1,l2)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=True)
for i, data in enumerate(train_loader, 0):
    #print(data[0].shape,data[1])
    #cv2.imshow('img',data[0])
    print(data[0].shape,data[1])
    img=data[0][0].numpy()*255
    img=img.astype('uint8')
    img=np.transpose(img,(1,2,0))
    img=img[:,:,::-1]
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        continue
    cv2.waitKey()
'''
