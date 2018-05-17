#!/bin/sh

#CUDA_VISIBLE_DEVICE=0 python3 trainGAN.py --lr 3e-4 --beta 3e-4 --batch-size 80 --epochs 200 --model-id 3 --comment model3

CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 1e-3 --beta 1e-5 --batch-size 128 --epochs 200 --model-id 1 --comment lre-3be-5bs128 &
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 1e-3 --beta 1e-4 --batch-size 128 --epochs 200 --model-id 1 --comment lre-3be-4bs128 &
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 1e-3 --beta 1e-3 --batch-size 128 --epochs 200 --model-id 1 --comment lre-3be-3bs128 &
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 1e-3 --beta 1e-2 --batch-size 128 --epochs 200 --model-id 1 --comment lre-3be-2bs128 & 
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 1e-3 --beta 1e-1 --batch-size 128 --epochs 200 --model-id 1 --comment lre-3be-1bs128 

wait

CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 1e-4 --beta 1e-5 --batch-size 128 --epochs 200 --model-id 1 --comment lre-4be-5bs128 &
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 1e-4 --beta 1e-4 --batch-size 128 --epochs 200 --model-id 1 --comment lre-4be-4bs128 &
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 1e-4 --beta 1e-3 --batch-size 128 --epochs 200 --model-id 1 --comment lre-4be-3bs128 &
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 1e-4 --beta 1e-2 --batch-size 128 --epochs 200 --model-id 1 --comment lre-4be-2bs128 & 
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 1e-4 --beta 1e-1 --batch-size 128 --epochs 200 --model-id 1 --comment lre-4be-1bs128 

wait

CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 5e-3 --beta 1e-5 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-3be-5bs128 &
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 5e-3 --beta 1e-4 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-3be-4bs128 &
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 5e-3 --beta 1e-3 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-3be-3bs128 &
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 5e-3 --beta 1e-2 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-3be-2bs128 & 
CUDA_VISIBLE_DEVICES=0 python3 trainGAN.py --lr 5e-3 --beta 1e-1 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-3be-1bs128 

wait

CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 5e-4 --beta 1e-5 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-4be-5bs128 &
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 5e-4 --beta 1e-4 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-4be-4bs128 &
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 5e-4 --beta 1e-3 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-4be-3bs128 &
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 5e-4 --beta 1e-2 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-4be-2bs128 & 
CUDA_VISIBLE_DEVICES=1 python3 trainGAN.py --lr 5e-4 --beta 1e-1 --batch-size 128 --epochs 200 --model-id 1 --comment lr5e-4be-1bs128
