#!/bin/bash

export root=$(pwd)/Miniconda3
export PATH=$root/Scripts:$root/Library/bin:$root/Library/mingw-w64/bin:$root/Library/usr/bin:$PATH

source $root/Scripts/activate $root

conda activate yoga

$root/envs/yoga/python.exe gui.py