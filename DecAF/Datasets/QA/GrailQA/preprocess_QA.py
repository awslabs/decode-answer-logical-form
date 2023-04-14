# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
import argparse
from collections import defaultdict

DATA_DIR = os.environ['DATA_DIR']

parser = argparse.ArgumentParser(description='process dataset')
parser.add_argument('--input_dir', type=str, default=f'{DATA_DIR}/tasks/QA/GrailQA_v1.0/raw')
parser.add_argument('--output_dir', type=str, default=f'{DATA_DIR}/tasks/QA/GrailQA_v1.0')
args = parser.parse_args()

if __name__ == "__main__":
    data_dict = defaultdict(list)
    for split in ['train', 'dev']:
        infname = os.path.join(args.input_dir, f"grailqa_v1.0_{split}.json")
        data = json.load(open(infname))
        for question in data:
            q_obj = {
                "QuestionId": question["qid"],
                "Question": question["question"],
                "Answers": [
                    {"freebaseId": answer["answer_argument"],
                    "text": answer["entity_name"] if "entity_name" in answer else answer["answer_argument"]} for answer in question["answer"]
                ]
            }
            data_dict[split].append(q_obj)

    infname = os.path.join(args.input_dir, f"grailqa_v1.0_test_public.json")
    data = json.load(open(infname))
    for question in data:
        q_obj = {
            "QuestionId": question["qid"],
            "Question": question["question"],
            "Answers": [
                {"freebaseId": "None",
                    "text": "None" } 
            ]
        }
        data_dict["test"].append(q_obj)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        print(len(data_dict[split]))
        outfname = os.path.join(args.output_dir, f"{split}.json")
        with open(outfname, 'w') as wf:
            json.dump(data_dict[split], wf, indent=2)