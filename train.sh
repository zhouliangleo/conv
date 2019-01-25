net="resnet34"
#net="resnet50"
#net="resnet152"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name ${net}
