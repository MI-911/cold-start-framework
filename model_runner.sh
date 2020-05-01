#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Must contain exactly one model name.";
    exit 1
fi
MODELNAME=$@
singularity instance start -B .:/app coldstart4.sigm $MODELNAME interview.py --include $MODELNAME

