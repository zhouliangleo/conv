# -*- coding:utf-8 -*-
import torch
import os
import random
import numpy as np
from PIL import Image
from model_class import get_class
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def write_result(vt,srcfile,model_ft):
    loc_file=os.path.join('.','split_person',vt,srcfile)
    video_file=os.path.join('/home/dl215/workspace/VIRAT/videos_original',srcfile.split('-')[0]+'.mp4')
    cap=cv2.VideoCapture(video_file)
    #f=open(loc_file,'r')
    lines=np.loadtxt(loc_file,delimiter=',')
    resultname='split_person/'+vt+'_result/'+srcfile
    fw=open(resultname,'w')
    for line in lines:
        line_split=line
        frameid=line_split[0]
        x1=int(float(line_split[2]))
        y1=int(float(line_split[3]))
        x2=int(float(line_split[4]))+x1
        y2=int(float(line_split[5]))+y1
        cap.set(1,int(frameid))
        _,frame=cap.read()
        frame=frame[y1:y2,x1:x2]
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  
        preds,p=get_class(frame,model_ft)
        fw.write('%d %s \n'%(preds,p))



if __name__=='__main__':
    model_ft=torch.load('model/best_model_res18_class4.pkl')
    for vt in ['val','test']:
        filenames=os.listdir('./split_person/'+vt)
        for file in filenames:
            print(vt+'/'+file)
            write_result(vt,file,model_ft)
