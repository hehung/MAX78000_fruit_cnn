#!/bin/sh
python train.py --model ai85net_fruit --dataset fruit --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai8x-fruit-qat8-q.pth.tar -8 --device MAX78000 "$@"