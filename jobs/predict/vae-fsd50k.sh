#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -o qlogs/
#$ -e qlogs/

#$ -l rocky
# -l centos

#$ -l cluster=andrena
#$ -l h_vmem=7.5G
#$ -pe smp 12

# -l node_type=rdg
# -l gpuhighmem
# -l h_vmem=20G
# -pe smp 12

rm -rf ~/.triton/cache
mamba activate perm
module load gcc
python -m synth_setter.cli.eval \
    experiment=surge/wandb_checkpoint/vae_full \
    paths.log_dir=/data/EECS-C4DM-Fazekas/benhayes/surge-preds/vae_fsd/ \
    datamodule=fsd \
    callbacks=eval_surge \
    mode=predict \
    datamodule.batch_size=1024 \
    datamodule.num_workers=11
