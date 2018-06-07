#!/bin/bash
wget https://github.com/dying1020/ML2018SPRING-file/releases/download/v1.0/ensemble.h5 -O model/ensemble.h5
python3 hw5_test.py $1 $2 model/tokenizer.pickle model/ensemble.h5