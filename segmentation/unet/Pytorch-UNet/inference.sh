#!/bin/bash
tgt="../target/"
src="../garfield_data_original/"
for file in ../garfield_data_original/*
do
  python predict.py -i ${src}${file} -o ${tg}${file} --model ./checkpoints/CP_epoch45.pth
done

