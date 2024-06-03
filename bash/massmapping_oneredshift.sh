#!/bin/bash

method=$1
Nsigma=4
idx_redshift=$2
ninpimgs=25
Nrea=25
batch_size=9

python ./scripts/massmapping.py $method ${method}_20240530_idx${idx_redshift}.pred --Nsigma $Nsigma \
        --idx-redshift $idx_redshift --ninpimgs $ninpimgs -v
python ./scripts/massmapping.py $method ${method}_20240530_idx${idx_redshift}.uq --Nsigma $Nsigma \
        --idx-redshift $idx_redshift --ninpimgs $ninpimgs \
        -b $batch_size --uq --nsamples $Nrea -v