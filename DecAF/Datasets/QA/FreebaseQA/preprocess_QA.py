# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
from collections import defaultdict, Counter
import argparse

DATA_DIR = os.environ['DATA_DIR']

# read data
def get_answers(question):
    """extract unique answers from question parses."""
    answers = set()
    for parse in question["Parses"]:
        for answer in parse["Answers"]:
            assert len(answer["AnswersName"]) == 1
            answers.add((answer["AnswersMid"],
                answer["AnswersName"][0]))
    return answers

def get_entities(question):
    """extract oracle entities from question parses."""
    entities = set()
    for parse in question["Parses"]:
        if parse["TopicEntityMid"] is not None:
            entities.add((parse["TopicEntityMid"], parse["TopicEntityName"]))
    return entities

parser = argparse.ArgumentParser(description='process dataset')
parser.add_argument('--input_dir', type=str, default=f'{DATA_DIR}/tasks/QA/FreebaseQA/raw')
parser.add_argument('--output_dir', type=str, default=f'{DATA_DIR}/tasks/QA/FreebaseQA')
args = parser.parse_args()

if __name__ == "__main__":
    questions_list = defaultdict(list)
    for split in ['train', 'dev', 'test']:
        num_answer_list = []
        num_without_answers = 0
        if split == 'test':
            infname = os.path.join(args.input_dir, f"FreebaseQA-eval.json")
        else:
            infname = os.path.join(args.input_dir, f"FreebaseQA-{split}.json")
        data = json.load(open(infname))
        for question in data["Questions"]:
            q_obj = {
                "QuestionId": question["Question-ID"],
                "Question": question["RawQuestion"],
                "Answers": [
                    {"freebaseId": answer[0],
                    "text": answer[1]}
                    for answer in get_answers(question)
                ]
            }
            num_answer_list.append(len(q_obj["Answers"]))
            # for answer in get_answers(question):
            #     if not answer[0].startswith("m."):
            #         print(answer)
            if len(get_answers(question)) == 0:
                num_without_answers += 1
    
            questions_list[f"{split}"].append(q_obj)
        print("num_without_answers: ", num_without_answers)
        print("answer num distribution: ", Counter(num_answer_list))

    for key in questions_list:
        print(key, len(questions_list[key]))
    
    # write down the processed dataset
    os.makedirs(args.output_dir, exist_ok=True)
    for key in ["train", "dev", "test"]:
        outfname = os.path.join(args.output_dir, f"{key}.json")
        with open(outfname, 'w') as wf:
            json.dump(questions_list[key], wf, indent=2)

