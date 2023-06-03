#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/Miniconda3

export root=$PWD/Miniconda3
export PATH=$root/Scripts:$root/Library/bin:$root/Library/mingw-w64/bin:$root/Library/usr/bin:$PATH

conda create -n yoga python=3.10 --yes
conda activate yoga
pip install -r requirements.txt

# Path: yoga-pose-recognition\run.sh