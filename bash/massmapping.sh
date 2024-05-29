#!/bin/bash

method=$1
Nsigma=4
ninpimgs=25
Nrea=25
batch_size=9

python ./scripts/massmapping.py $method ${method}_20240528.pred --Nsigma $Nsigma --ninpimgs $ninpimgs -v
python ./scripts/massmapping.py $method ${method}_20240528.uq --Nsigma $Nsigma --ninpimgs $ninpimgs \
        -b $batch_size --uq --nsamples $Nrea -v