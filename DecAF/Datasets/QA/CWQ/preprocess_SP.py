# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
from collections import Counter
import argparse
from DecAF.Preprocess.linearize import load_nameid_dict
from DecAF.Datasets.QA.utils import execute_vanilla_s_expr, revise_only_name
from collections import defaultdict


DATA_DIR = os.environ['DATA_DIR']

parser = argparse.ArgumentParser(description='process dataset')
parser.add_argument('--data_dir', type=str, default=f'{DATA_DIR}/tasks/QA/CWQ')
args = parser.parse_args()


if __name__ == "__main__":
    name_dir = f"{DATA_DIR}/knowledge_source/Freebase/id2name_parts_disamb"
    name2id_dict, id2name_dict = load_nameid_dict(name_dir, lower=False)

    # Only keep the training data with correct excution results
    if not os.path.exists(os.path.join(args.data_dir, "raw/ComplexWebQuestions_train_filtered.expr.json")):
        print("filtering the training data")
        data_file = os.path.join(args.data_dir, "raw/ComplexWebQuestions_train.expr.json")
        with open(data_file, "r") as rf:
            data = json.load(rf)
        new_data = []
        for data_i in data:
            answers = [answer_i["answer_id"] for answer_i in data_i["answers"]]
            parsing_results = execute_vanilla_s_expr(data_i["SExpr"])
            if set(answers) == set(parsing_results):
                new_data.append(data_i)
        print("remaining rate: ", len(new_data) / len(data))
        save_file = os.path.join(args.data_dir, "raw/ComplexWebQuestions_train_filtered.expr.json")
        with open(save_file, "w") as wf:
            json.dump(new_data, wf, indent=2)

    # preprocess the s-expression
    num_s_expr = []
    data_dict = defaultdict(list)
    none_answer_dict = defaultdict(int)

    ID2LFs = defaultdict(list)

    for split in ['train', 'dev', 'test']:
        if split == "train":
            infname = os.path.join(args.data_dir, f"raw/ComplexWebQuestions_{split}_filtered.expr.json")
        else:
            infname = os.path.join(args.data_dir, f"raw/ComplexWebQuestions_{split}.expr.json")
        data = json.load(open(infname))

        for question in data:
            s_express_list = [question["SExpr"]] if question["SExpr"] != "null" else []
            num_s_expr.append(len(s_express_list))
            if len(s_express_list) == 0:
                none_answer_dict[split] += 1
                if split == "train":
                    continue
                LFs = {}
            else:
                LFs = {
                    "LF_original": s_express_list,
                    "LF_processed": [revise_only_name(s_expr, id2name_dict) for s_expr in s_express_list],
                } 
            ID2LFs[question["ID"]] = LFs

    print(Counter(num_s_expr))

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
        with open(outfname, 'w') as f:
            json.dump(data_dict[split], f, indent=2)


