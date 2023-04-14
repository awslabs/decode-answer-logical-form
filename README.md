# DecAF: Joint Decoding Answer and Logical Form for KBQA through Retrieval

This is the official implementation of our paper "DecAF: Joint Decoding Answer and Logical Form for Knowledge Base Question Answering through Retrieval", in ICLR 2023 [OpenReview](https://openreview.net/forum?id=hHMm9YVW0xz) [arXiv](https://arxiv.org/abs/2210.00063).

## 1. Set up 

```
conda create -n DecAF python=3.8
conda activate DecAF
pip install -r requirements.txt
pip install -e .
source config.sh ${your base directory to store data, models, and results}
conda activate DecAF
```
Then follow instructions in https://github.com/dki-lab/Freebase-Setup to set up the Freebase server, which is necessary for data preprocessing and evaluation.

## 2. Preprocessing

### Knowledge source

Download the [Freebase](https://drive.google.com/file/d/13y_mwHI3pyARqIfjMxyU17U-aC0eQKHB/view?usp=share_link) data, extract and put it under `${DATA_DIR}/knowledge_source/Freebase`.
The directory structure should be like this:
```
${DATA_DIR}/knowledge_source/Freebase
├── topic_entities_parts
├── triple_edges_parts
├── id2name_parts
├── id2name_parts_disamb
```

Preprocess the knowledge source, i.e., Freebase:
```
python DecAF/Knowledge/process_freebase.py --data_dir ${DATA_DIR}/knowledge_source/Freebase
```

### Datasets

Please see `DecAF/Datasets` for preprocessing datasets including WebQSP, GrailQA, ComplexWebQuestions, and FreebaseQA.

## 3. Retrieval

We use [PySerini](https://github.com/castorini/pyserini) for BM25 retrieval.
```
cd DecAF/Retrieval/Pyserini
bash build_index_sparse.sh      # build sparse index for knowledge source
bash run_search_sparse.sh -d GrailQA -s dev     # retrieve from knowledge source
```
You can change `-d` argument to WebQSP, CWQ, or FreebaseQA, and `-s` argument to train or test.

You should see the following results:
|            | WebQSP (test) | CWQ (test) | GrailQA (dev) | FreebaseQA (test) |
|------------|--------|-----|---------|------------|
| Hits@100   | 81.3      | 63.4   | 90.1    | 93.6          |
| Recall@100 | 67.8      | 57.5   | 85.0    | 93.6          |

if you encounter errors about java version, try to install [java 11](https://download.oracle.com/java/17/archive/jdk-17.0.3.1_linux-x64_bin.tar.gz) and run:
```
export JAVA_HOME={YOUR_OWN_PATH}/jdk-17.0.3.1
```

For dense retrieval, we use [DPR](https://github.com/facebookresearch/DPR) and train it on each dataset. We refer the readers to the original repo for details on how to conduct training and inference with DPR. It should be emphasized that, based on our experiments, DPR demonstrates superior performance in comparison to BM25 solely on the WebQSP dataset.

## 4. Reading (Answer Generation)

We use [FiD](https://github.com/facebookresearch/FiD) as the reading module, which takes the output of the retriving module as input. Note that FiD requires transformers==3.0.2, which is conflicting with the version required by PySerini. We recommend to create a new conda environment for FiD. Remember to run `source config.sh ${your base directory to store data, models, and results}` again after creating the new environment to set the environment variables.

Process the retrieval results to the format required by FiD:
```
cd DecAF/Reading
python process_fid.py --retrieval_data_path ${SAVE_DIR}/Retrieval/pyserini/search_results/QA_GrailQA_Freebase_BM25 --mode SPQA
```
You can change the `--mode` argument to QA, which is for FreebaseQA since it does not provide anotated logical forms.

Download FiD and replace it with our modified code which supports beam search:
```
git clone https://github.com/facebookresearch/FiD.git

cp test_reader.py FiD/ 
cp train_reader.py FiD/
```

Train FiD:
```
bash bash/run_train.sh -t ${SAVE_DIR}/Retrieval/pyserini/search_results/QA_GrailQA_Freebase_BM25/train_fid_SPQA.json -e ${SAVE_DIR}/Retrieval/pyserini/search_results/QA_GrailQA_Freebase_BM25/dev_fid_SPQA.json -s 30000 
```
`-t` argument is the path of the training data, `-e` argument is the path of the evaluation data, and `-s` argument is the number of training steps. We recommend to use 30000 steps for WebQSP, GrailQA, and FreebaseQA while 60000 steps for CWQ.

Inference with FiD:
```
bash bash/run_test.sh -d ${SAVE_DIR}/Retrieval/pyserini/search_results/QA_WebQSP_Freebase/test_fid_SPQA.json -m ${MODEL_DIR}/Reading/FiD/WebQSP_Freebase_DPR_FiDlarge -b 10
```
`-d` argument is the path of the inference queries, `-m` argument is the path of the trained model, and `-b` argument is the beam size. We recommend to use 10 for WebQSP and GrailQA, 1 for FreebaseQA while 20 for CWQ.


## 5. Evaluation

```
cd DecAF/Datasets/QA

python evaluate.py --dataset GrailQA --result_path ${SAVE_DIR}/Reading/FiD/QA_GrailQA_Freebase_BM25_large_100_p100_b10/final_output_dev_fid_SPQA.json
```     
You can change the `--dataset` argument to WebQSP, CWQ, or FreebaseQA, and `--result_path` argument to the path of the inference results.


## 6. Pre-trained Models and Predicted Results

| FiD Model      | Prediction | Metric |
|------------|------------|--------|
| QA_WebQSP_Freebase_BM25_large_100         | [p100_b10 (test)](https://drive.google.com/drive/folders/1GSGKeeVifIZmFiaf3EM2VxdtSs6cJKU1?usp=share_link)         | F1=75.3      |
| QA_WebQSP_Freebase_DPR_large_100         | [p100_b10 (test)](https://drive.google.com/drive/folders/15mON6TNZeAqDZ5Vvt2i7Mb5B6GAKKWsv?usp=share_link)          | F1=77.1      |
| QA_WebQSP_Freebase_DPR_3b_100         | [p100_b15 (test)](https://drive.google.com/drive/folders/1Z5UBByFsAaRxteORn0HIr62oRWZs08gW?usp=share_link)          | F1=78.8      |
| QA_GrailQA_Freebase_BM25_large_100 | [p100_b10 (dev)](https://drive.google.com/drive/folders/1uVwgpFLBBvdNeHFYiYKPFL6U9GfgLW63?usp=share_link)          | F1=78.7      |
| QA_GrailQA_Freebase_BM25_3b_100 | [p100_b15 (dev)](https://drive.google.com/drive/folders/1cGAMhu9vb067BbUBe2hjIFUYuQMYihvw?usp=share_link)          | F1=81.4      |
| QA_CWQ_Freebase_BM25_large_100 | [p100_b20 (test)](https://drive.google.com/drive/folders/1-MJtGTCfCm21EhWnfbxBkM1piSIAMub7?usp=share_link)          | Hits@1=68.7      |
| QA_CWQ_Freebase_BM25_3b_100 | [p100_b15 (test)](https://drive.google.com/drive/folders/1rXmbd0Atn0k4of--5lDgUvQjDYE_yb_y?usp=share_link)          | Hits@1=70.4      |
| QA_FreebaseQA_Freebase_BM25_large_100 | [p100_b1 (test)](https://drive.google.com/drive/folders/1veeGlFCOYMjx6SriADJ-rGT7Xiu0C6Nt?usp=share_link)          | Hits@1=80.6      |