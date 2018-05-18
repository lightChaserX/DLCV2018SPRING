#!/bin/sh

echo test VAE
python3 testVAE.py --data-path $1 --dir $2 --batch-size 1000

echo test GAN
python3 testGAN.py --dir $2

echo test ACGAN
python3 testACGAN.py --dir $2