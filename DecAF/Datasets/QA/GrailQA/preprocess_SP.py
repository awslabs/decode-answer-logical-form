# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
import argparse
from DecAF.Preprocess.linearize import load_nameid_dict
from DecAF.Datasets.QA.utils import revise_only_name
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)


DATA_DIR = os.environ['DATA_DIR']

parser = argparse.ArgumentParser(description='process dataset')
parser.add_argument('--data_dir', type=str, default=f'{DATA_DIR}/tasks/QA/GrailQA')
args = parser.parse_args()

if __name__ == "__main__":
    name_dir = f"{DATA_DIR}/knowledge_source/Freebase/id2name_parts_disamb"
    name2id_dict, id2name_dict = load_nameid_dict(name_dir, lower=False)

    data_dict = defaultdict(list)
    ID2LFs = defaultdict(list)
    for split in ['train', 'dev']:
        infname = os.path.join(args.data_dir, f"raw/grailqa_v1.0_{split}.json")
        data = json.load(open(infname))
        for question in data:
            assert isinstance(question["s_expression"], str)
            LFs = {
                    "LF_original": [question["s_expression"]],
                    "LF_processed": [revise_only_name(question["s_expression"], id2name_dict)],
                } 
            ID2LFs[question["qid"]] = LFs
    
    for split in ['train', 'dev', 'test']:
        infname = os.path.join(args.data_dir, f"{split}.json")
        data = json.load(open(infname))

        for question in data:
            question_id = question["QuestionId"]
            question.update(ID2LFs[question_id])
            data_dict[split].append(question)

    # write down the processed dataset
    os.makedirs(args.data_dir, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        print(len(data_dict[split]))
        outfname = os.path.join(args.data_dir, f"{split}.json")
        with open(outfname, 'w') as wf:
            json.dump(data_dict[split], wf, indent=2)


