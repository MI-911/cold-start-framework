#!/bin/bash
tail -f ~/.singularity/instances/logs/$SLURM_JOB_NODELIST/$SLURM_JOB_USER/$@.out
