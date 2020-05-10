#!/usr/bin/env bash
#SBATCH --job-name MindReader # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --mail-type=ALL # NONE, BEGIN, END, FAIL, REQUEUE, ALL TIME_LIMIT, TIME_LIMIT_90, etc
#SBATCH --mail-user=tjenda15@student.aau.dk # CHANGE THIS to your email address!
#SBATCH --qos=allgpus # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-03.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
#SBATCH --error=job.mindreader.err
#SBATCH --output=job.mindreader.out
#SBATCH --cpus-per-task=6
srun --pty singularity run --nv -B ..:/app ../coldstart.sigm $@
