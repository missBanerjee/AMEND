#!/bin/bash

set -e
set -x
SAVE_DIR=/home/anwesha/SimGCD/osr_novel_categories/cifar100/
mkdir -p $SAVE_DIR
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m model.AMEND \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --grad_from_block 11 \
    --aug_epochs 0\
    --nearest_epochs 200\
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --lam 1\
    --neighbors 5\
    --expanded_neighbor 4\
    --feat_bank_size 2048\
    --exp_name cifar100_expanded_neg_margin
> ${SAVE_DIR}log_scars_${EXP_NUM}.out
