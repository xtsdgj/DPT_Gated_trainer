#!/bin/bash

#conda activate tc1131

# 循环运行 10000 次
for (( i=1; i<=10; i++ ))
do
   echo "Running iteration: $i"
   python ./src/train.py /home/xt/PycharmProjects/trianingModule/src/configs/args_train_example.txt
done
