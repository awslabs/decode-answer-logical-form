#!/bin/bash

usage()
{
    cat << EOF
usage: $0 options
OPTIONS:
        -h      Show the help and exit
        -d      data path
        -m      model path
        -b      number of beams
EOF
}

while getopts "h:d:m:b:" opt
do
    case $opt in
        h)
            usage
            exit 1
            ;;
        d)
            eval_data=$OPTARG
            ;;
        m)
            model_path=$OPTARG
            ;;
        b)
            num_beams=$OPTARG
            ;;
    esac
done

n_context=100

model_name=$(echo "$model_path" | awk -F'/' '{print $NF}')

N_GPU=2
export NGPU=${N_GPU}
cmd="CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch 
        --nproc_per_node=${N_GPU} --master_port 1235 
        FiD/test_reader.py 
        --model_path ${model_path}  
        --eval_data ${eval_data}   
        --per_gpu_batch_size 1 
        --num_beams ${num_beams} 
        --text_maxlength 200 
        --answer_maxlength 128 
        --n_context ${n_context} 
        --name ${model_name}_p${n_context}_b${num_beams} 
        --checkpoint_dir ${SAVE_DIR}/Reading/FiD 
        --write_results"

echo $cmd
eval $cmd