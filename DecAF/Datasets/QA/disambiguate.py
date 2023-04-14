# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import json
from tqdm import tqdm
import argparse
from DecAF.Preprocess.linearize import load_nameid_dict
import logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.environ['DATA_DIR']

parser = argparse.ArgumentParser(description='disambiguate dataset answer entities')
parser.add_argument('--data_dir', type=str, default=f'{DATA_DIR}/tasks/QA/CWQ')
args = parser.parse_args()


if __name__ == "__main__":
    # load id2name mapping
    logging.info("Loading id2name mapping")
    name_dir = f"{DATA_DIR}/knowledge_source/Freebase/id2name_parts_disamb"
    name2id_dict, id2name_dict = load_nameid_dict(name_dir, lower=False)

    # load original QA dataset
    for split in ["dev", "test", "train"]:
        if f"{split}.json" not in os.listdir(args.data_dir):
            continue
        org_file_path = os.path.join(args.data_dir, f"{split}.json")
        with open(org_file_path, "r") as f:
            org_data = json.load(f)

        new_data = []
        for item in tqdm(org_data):
            new_answers = []
            for answer in item["Answers"]:
                if answer["freebaseId"] in id2name_dict:
                    new_answers.append({
                        "freebaseId": answer["freebaseId"],
                        "text": id2name_dict[answer["freebaseId"]],
                    })
                else:
                    new_answers.append(answer)
            new_data.append({
                "QuestionId": item["QuestionId"],
                "Question": item["Question"],
                "Answers": new_answers,
            })

        save_file_path = os.path.join(args.data_dir, f"{split}.json")
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        with open(save_file_path, "w") as f:
            json.dump(new_data, f, indent=2)