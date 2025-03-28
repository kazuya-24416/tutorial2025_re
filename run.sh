#!/bin/bash

#$-j y
#$-m a
#$-m b
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro/4.1.2 cuda/12.2/12.2.0 cudnn/8.9/8.9.7 nccl/2.18/2.18.5-1
cd /home/acg16612ik/tutorial2025_re
export CUDA_HOME=/usr/local/cuda
singularity exec --bind ./:/workspace --bind /groups/gce50999/acg16612ik:/groups --nv ./llm_abci.sif python src/train_seq2seq.py
