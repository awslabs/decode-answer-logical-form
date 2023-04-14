# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
from DecAF.Datasets.QA.utils import is_valid

DATA_DIR = os.environ['DATA_DIR']

parser = argparse.ArgumentParser(description='process CWQ dataset')
parser.add_argument('--input_dir', type=str, default=f'{DATA_DIR}/tasks/QA/CWQ/raw')
parser.add_argument('--output_dir', type=str, default=f'{DATA_DIR}/tasks/QA/CWQ')
args = parser.parse_args()

if __name__ == "__main__":
    # process the test data so that each answer contains the "aliases" key
    # this is just for the requirement of the evaluation script
    infname = os.path.join(args.input_dir, "ComplexWebQuestions_test.json")
    data = json.load(open(infname, "r"))
    for question in data:
        for answer in question["answers"]:
            if "aliases" not in answer:
                answer["aliases"] = []
    json.dump(data, open(infname, "w"), indent=2)

    # process the train/dev/test data
    questions_list = defaultdict(list)
    for split in ['train', 'dev', 'test']:
        infname = os.path.join(args.input_dir, f"ComplexWebQuestions_{split}.json")
        data = json.load(open(infname))
        for question in data:
            q_obj = {
                "QuestionId": question["ID"],
                "Question": question["question"],
                "Answers": [
                    {"freebaseId": answer["answer_id"],
                    "text": answer["answer"]}
                    for answer in question["answers"]
                ]
            }
            questions_list[split].append(q_obj)
    
    # write down the processed dataset
    os.makedirs(args.output_dir, exist_ok=True)
    for key in ['train', 'dev', 'test']:
        print(f"{key} has {len(questions_list[key])} questions")
        outfname = os.path.join(args.output_dir, f"{key}.json")
        with open(outfname, 'w') as wf:
            json.dump(questions_list[key], wf, indent=2)