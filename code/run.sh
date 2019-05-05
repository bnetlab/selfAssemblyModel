#!/bin/bash

source activate tensorflow

python integral.5.py 4 -25.0 25.0 0.1 1 1
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1 1 1 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1.5 1 1 &
	CUDA_DEVICE=3 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2 1 1 &
	CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2.5 1 1 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3 1 1 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3.5 1 1 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 4 1 1
wait
done
echo "1 done"

python integral.5.py 4 -25.0 25.0 0.1 2 1
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1 2 1 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1.5 2 1 &
	CUDA_DEVICE=3 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2 2 1 &
	CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2.5 2 1 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3 2 1 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3.5 2 1 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 4 2 1
wait
done
echo "2 done"

python integral.5.py 4 -25.0 25.0 0.1 4 1
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1 4 1 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1.5 4 1 &
	CUDA_DEVICE=3 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2 4 1 &
	CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2.5 4 1 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3 4 1 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3.5 4 1 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 4 4 1
wait
done
echo "3 done"

python integral.5.py 4 -25.0 25.0 0.1 2 2
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1 2 2 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1.5 2 2 &
	CUDA_DEVICE=3 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2 2 2 &
	CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2.5 2 2 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3 2 2 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3.5 2 2 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 4 2 2
wait
done
echo "4 done"

python integral.5.py 4 -25.0 25.0 0.1 4 2
for i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1 4 2 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1.5 4 2 &
	CUDA_DEVICE=3 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2 4 2 &
	CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2.5 4 2 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3 4 2 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3.5 4 2 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 4 4 2
wait
done
echo "5 done"

python integral.5.py 4 -25.0 25.0 0.1 4 4
ffor i in 1 1.5 2 2.5 3 3.5 4
do
	CUDA_DEVICE=1 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1 4 4 &
	CUDA_DEVICE=2 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 1.5 4 4 &
	CUDA_DEVICE=3 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2 4 4 &
	CUDA_DEVICE=4 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 2.5 4 4 &
	CUDA_DEVICE=5 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3 4 4 & 
	CUDA_DEVICE=6 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 3.5 4 4 &
	CUDA_DEVICE=7 python cdua_mainfn_4m.py 0.1 0 15 -10 $i 4 4 4
wait
done
echo "6 done"

