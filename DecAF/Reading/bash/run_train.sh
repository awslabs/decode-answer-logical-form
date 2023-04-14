#!/bin/bash

usage()
{
    cat << EOF
usage: $0 options
OPTIONS:
        -h      Show the help and exit
        -t      training data path
        -e      evaluation data path
        -s      total steps
EOF
}

while getopts "h:t:e:s:" opt
do
    case $opt in
        h)
            usage
            exit 1
            ;;
        t)
            train_data=$OPTARG
            ;;
        e)
            eval_data=$OPTARG
            ;;
        s)
            total_step=$OPTARG
            ;;
    esac
done

n_context=100
answer_maxlength=128
model_size="large"
lr=5e-5

model_base=$(echo "$train_data" | awk -F'/' '{print $(NF-1)}')
model_name="${model_base}_${model_size}_${n_context}"

N_GPU=2
export NGPU=${N_GPU}
cmd="CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch  
        --nproc_per_node=${N_GPU} --master_port 1237  
        FiD/train_reader.py 
        --train_data ${train_data}
        --eval_data ${eval_data}  
        --model_size ${model_size}  
        --name ${model_name}  
        --checkpoint_dir ${MODEL_DIR}/Reading/FiD 
        --use_checkpoint
        --lr ${lr} 
        --optim adamw 
        --scheduler linear 
        --weight_decay 0.01 
        --text_maxlength 200 
        --answer_maxlength ${answer_maxlength} 
        --per_gpu_batch_size 1 
        --total_batch_size 16 
        --n_context ${n_context} 
        --total_step ${total_step} 
        --scheduler_steps ${total_step}
        --warmup_step 1000
        --eval_freq 2000
        --save_freq ${total_step}"

echo $cmd
eval $cmd