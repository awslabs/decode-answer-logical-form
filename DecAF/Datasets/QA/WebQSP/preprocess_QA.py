# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
from collections import defaultdict
import argparse

DATA_DIR = os.environ['DATA_DIR']

def get_answers(question):
    """extract unique answers from question parses."""
    answers = set()
    for parse in question["Parses"]:
        for answer in parse["Answers"]:
            answers.add((answer["AnswerArgument"],
                answer["EntityName"]))
    return answers

def get_entities(question):
    """extract oracle entities from question parses."""
    entities = set()
    for parse in question["Parses"]:
        if parse["TopicEntityMid"] is not None:
            entities.add((parse["TopicEntityMid"], parse["TopicEntityName"]))
    return entities

parser = argparse.ArgumentParser(description='process WebQSP dataset')
parser.add_argument('--input_dir', type=str, default=f'{DATA_DIR}/tasks/QA/WebQSP/raw/data')
parser.add_argument('--output_dir', type=str, default=f'{DATA_DIR}/tasks/QA/WebQSP')
args = parser.parse_args()

if __name__ == "__main__":
    questions_list = defaultdict(list)
    for split in ['train', 'test']:
        num_without_answers = 0
        infname = os.path.join(args.input_dir, f"WebQSP.{split}.json")
        data = json.load(open(infname))
        for question in data["Questions"]:
            q_obj = {
                "QuestionId": question["QuestionId"],
                "Question": question["ProcessedQuestion"],
                "Answers": [
                    {"freebaseId": answer[0],
                    "text": answer[1]}
                    for answer in get_answers(question)
                ]
            }
            if len(get_answers(question)) == 0:
                num_without_answers += 1

            questions_list[split].append(q_obj)
        print(num_without_answers)
    
    # write down the processed dataset
    os.makedirs(args.output_dir, exist_ok=True)
    for key in ["train", "test"]:
        print(f"{key} has {len(questions_list[key])} questions")
        outfname = os.path.join(args.output_dir, f"{key}.json")
        with open(outfname, 'w') as wf:
            json.dump(questions_list[key], wf, indent=2)

