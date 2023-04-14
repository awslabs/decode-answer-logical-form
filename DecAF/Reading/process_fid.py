# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
import argparse
from tqdm import tqdm
from DecAF.Datasets.QA.utils import parse_answer


parser = argparse.ArgumentParser(description='Process the retrieved data to fit the format of FiD model')
parser.add_argument("--retrieval_data_path", type=str, 
                    default='/home/ubuntu/data/KBQA/WebQSP/test.json')
parser.add_argument("--mode", type=str, 
                    default='SPQA', help="SPQA or QA or SP")
args = parser.parse_args()


for split in ["dev", "train", "test"]:

    file_path = os.path.join(args.retrieval_data_path, f"{split}.json")
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r") as rf:
        data = json.load(rf)

    new_data_qa = []
    new_data_sp = []
    for data_i in tqdm(data):
        if args.mode != "QA":
            if "LF_processed" in data_i:
                new_data_i = {
                    "id": str(data_i["QuestionId"]) + ":SP",
                    "question": "Semantic Parsing: " + data_i["Question"],
                    "answers": data_i["LF_processed"],
                    "ctxs": data_i["ctxs"],
                }
                new_data_sp.append(new_data_i)
            elif split != "train":
                new_data_i = {
                    "id": str(data_i["QuestionId"]) + ":SP",
                    "question": "Semantic Parsing: " + data_i["Question"],
                    "answers": ["none"],
                    "ctxs": data_i["ctxs"],
                }
                new_data_sp.append(new_data_i)
        if args.mode != "SP":
            new_data_i = {
                "id": str(data_i["QuestionId"]) + ":QA",
                "question": "Question Answering: " + data_i["Question"],
                "answers": parse_answer(data_i["Answers"]),
                "ctxs": data_i["ctxs"],
            }
            new_data_qa.append(new_data_i)
    new_data = new_data_qa + new_data_sp
    print(len(new_data))

    output_file = file_path.replace(".json", f"_fid_{args.mode}.json")
    print(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as wf:
        json.dump(new_data, wf, indent=2)