#!/bin/bash
# parameter : dim,xrange, yrange, bin, mu, lambda, tau, T
source activate tensorflow
python integral.5.py $1 $2 $3 $4 $5 $6
conda deactivate 
matlab -nodesktop -nosplash -r "mainfn($4,$2,$3,$7,$8); exit;"
