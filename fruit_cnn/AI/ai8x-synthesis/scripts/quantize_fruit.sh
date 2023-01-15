#!/bin/sh
python quantize.py trained/ai8x-fruit-qat8.pth.tar trained/ai8x-fruit-qat8-q.pth.tar --device MAX78000 -v "$@"