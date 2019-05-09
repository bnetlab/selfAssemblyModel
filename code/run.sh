#!/bin/bash

source activate tensorflow
#python getIndex.py 0.05 0 10 -10

#python integral.5.py 4 -20.0 20.0 0.05 1 1
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 0.5 1 1 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 0.75 1 1 &
	CUDA_DEVICE=0 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 1 1 1 &
	#CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 1.5 1 1 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 2 1 1 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 3 1 1 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 4 1 1
wait
done
echo "1 done"

python integral.5.py 4 -20.0 20.0 0.05 1 2
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 0.5 1 2 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 0.75 1 2 &
	CUDA_DEVICE=0 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 1 1 2 &
	#CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 1.5 1 2 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 2 1 2 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 3 1 2 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 4 1 2
wait
done
echo "2 done"

python integral.5.py 4 -20.0 20.0 0.05 1 4
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 0.5 1 4 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 0.75 1 4 &
	CUDA_DEVICE=0 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 1 1 4 &
	#CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 1.5 1 4 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 2 1 4 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 3 1 4 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.05 0 10 -10 $i 4 1 4
wait
done
echo "3 done"
