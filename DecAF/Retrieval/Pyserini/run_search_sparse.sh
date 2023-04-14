#!/bin/bash

usage()
{
    cat << EOF
usage: $0 options
OPTIONS:
        -h      Show the help and exit
        -d      dataset name: WebQSP, CWQ, GrailQA, FreebaseQA
        -s      split: train, dev, test
EOF
}

while getopts "h:d:s:" opt
do
    case $opt in
        h)
            usage
            exit 1
            ;;
        d)
            dataset=$OPTARG
            ;;
        s)
            split=$OPTARG
            ;;
    esac
done


index_name="Freebase"
task="QA" 
name="${task}_${dataset}_${index_name}_BM25"
output_dir="${SAVE_DIR}/Retrieval/pyserini/search_results/${name}"

python search.py \
    --query_data_path ${DATA_DIR}/tasks/${task}/${dataset}/${split}.json \
    --index_name ${index_name} \
    --output_dir ${output_dir} \
    --top_k 100 \
    --k1 0.4 \
    --b 0.4 \
    --num_process 10 \
    --num_queries -1 \
    --eval \
    --save
