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

source jobs/predict/get-ckpt-from-wandb.sh drk06g0w
echo "Using wandb directory: $WANDB_DIR"
echo "Using checkpoint: $CKPT_PATH"

rm -rf ~/.triton/cache
mamba activate perm
module load gcc
python src/eval.py \
    experiment=surge/flow_full \
    model.test_cfg_strength=2.0 \
    model.test_sample_steps=100 \
    paths.log_dir=/data/EECS-C4DM-Fazekas/benhayes/surge-preds/flow_full/ \
    data=surge \
    callbacks=eval_surge \
    mode=predict \
    data.batch_size=1024 \
    data.num_workers=11 \
    ckpt_path=$CKPT_PATH
