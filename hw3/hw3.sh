#!/bin/sh

python3 test.py --g $1 --p $2 --model-name FCN8s
python3 mean_iou_evaluate.py -g $1 -p $2