#!/bin/sh

echo model1
#python3 train.py --lr 1e-4 --kl 1e-4 --batch-size 1024 --epochs 200 --parallel True --model-id 1
echo model2
#python3 train.py --lr 1e-4 --kl 1e-2 --batch-size 1024 --epochs 200 --parallel True --model-id 2
echo model3
#python3 train.py --lr 1e-3 --kl 1e-4 --batch-size 1024 --epochs 200 --parallel True --model-id 3
echo model4 \(add layers to the decoder\)
#python3 train.py --lr 1e-3 --kl 1e-4 --batch-size 1024 --epochs 200 --parallel True --model-id 4
echo model5 \(new architecture\)
#python3 train.py --lr 1e-3 --kl 1e-5 --batch-size 512 --epochs 200 --parallel True --model-id 5
echo model6 \(size_avr = true\)
#python3 train.py --lr 1e-3 --kl 1e-5 --batch-size 512 --epochs 200 --parallel True --model-id 6
echo model7 \(size_avr = true\)
#python3 train.py --lr 1e-3 --kl 1e-5 --batch-size 512 --epochs 200 --parallel True --model-id 7
echo model8 \(batchsize 122\)
#CUDA_VISIBLE_DEVICE=0 python3 train.py --lr 1e-3 --kl 1e-5 --batch-size 32 --epochs 200 --model-id 8
echo model9
CUDA_VISIBLE_DEVICE=0 python3 train.py --lr 1e-3 --kl 1e-5 --batch-size 32 --epochs 200 --model-id 9
