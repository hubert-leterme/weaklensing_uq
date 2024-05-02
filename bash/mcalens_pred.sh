#!/bin/bash

method=mcalens
picklefile=mcalens.pred
ninpimgs=25
niter=32

python scripts/massmapping.py $method $picklefile --ninpimgs $ninpimgs --niter $niter -v