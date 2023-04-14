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
import argparse
from DecAF.Knowledge.linearize import (
    load_nameid_dict,
    get_raw_name,
)
from DecAF.Datasets.QA.utils import (
    Prefix_to_id_all, 
    answer_ensemble, 
    id_to_name,
    to_webqsp_format,
    to_grailqa_format,
    to_cwq_format,
    compute_hits1_dict,
)
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_DIR = os.environ['DATA_DIR']
SAVE_DIR = os.environ['SAVE_DIR']

parser = argparse.ArgumentParser(description='dataset evaluation')
parser.add_argument('--dataset', type=str, help="WebQSP, GrailQA, CWQ, FreebaseQA")
parser.add_argument('--result_path', type=str, help="result directory")
args = parser.parse_args()

result_path = args.result_path
if not os.path.exists(result_path):
    exit(f"{result_path} not found")

# the evaluation for FreebaseQA is different from the other three datasets
# since it does not provide the labeled logical forms
# we only evaluate the predicted answers under the QA setting
if args.dataset == "FreebaseQA":
    with open(result_path, "r") as rf:
        results = json.load(rf)
    gold_answers, pred_answers = {}, {}
    for key in results:
        gold_answers[key] = results[key]["gold answers"]
        gold_answers[key] = [get_raw_name(each) for each in gold_answers[key]]
        pred_answers[key] = results[key]["predicted answers"]
        pred_answers[key] = [get_raw_name(each) for each in pred_answers[key]]
    logging.info("Hits@1: {}".format(compute_hits1_dict(pred_answers, gold_answers)))
    exit()

name_dir = DATA_DIR + "/knowledge_source/Freebase/id2name_parts_disamb"
name2id_dict, id2name_dict = load_nameid_dict(name_dir, lower=False)

name_dir = DATA_DIR + "/knowledge_source/Freebase/id2name_parts"
name2id_dict_orig, id2name_dict_orig = load_nameid_dict(name_dir, lower=False)

logging.info("parse generation results to answers")
save_file_path_all = Prefix_to_id_all(result_path, name2id_dict)

logging.info("Combine answers from SP and QA")
save_file_path_id_final = answer_ensemble(save_file_path_all)

if args.dataset == "WebQSP":
    logging.info("Evaluation Results on WebQSP:")
    save_file_path_id_offical = to_webqsp_format(save_file_path_id_final)
    eval_code = os.path.join(DATA_DIR, "tasks/QA/WebQSP/raw/eval/eval.py")
    gold_file = os.path.join(DATA_DIR, "tasks/QA/WebQSP/raw/data/WebQSP.test.json")
    os.system(f"python2 {eval_code} {gold_file} {save_file_path_id_offical}")
elif args.dataset == "GrailQA":
    logging.info("Evaluation Results on GrailQA:")
    save_file_path_id_offical = to_grailqa_format(save_file_path_id_final)
    gold_file = os.path.join(DATA_DIR, f"tasks/QA/GrailQA/raw/grailqa_v1.0_dev.json")
    os.system(f"python GrailQA/evaluate.py {gold_file} {save_file_path_id_offical} --fb_roles SP_tools/ontology/fb_roles --fb_types SP_tools/ontology/fb_types --reverse_properties SP_tools/ontology/reverse_properties")
elif args.dataset == "CWQ":
    logging.info("Evaluation Results on CWQ:")
    save_file_path_name_final = id_to_name(save_file_path_id_final, id2name_dict_orig)
    save_file_path_name_official = to_cwq_format(save_file_path_name_final)
    gold_file = os.path.join(DATA_DIR, f"tasks/QA/CWQ/raw/ComplexWebQuestions_test.json")
    os.system(f"python CWQ/eval_script.py {gold_file} {save_file_path_name_official}")
