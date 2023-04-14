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


import random
import json
import itertools
from collections import defaultdict
from DecAF.Knowledge.linearize import get_raw_name
from collections import OrderedDict
from DecAF.Datasets.QA.SP_tools.executor.sparql_executor import get_label, execute_query
from DecAF.Datasets.QA.SP_tools.executor.logic_form_util import lisp_to_sparql
import re
from tqdm import tqdm
import time as time
import signal

#Close session
def handler(signum, frame):
    print("SPARQL Executation Time out!")
    raise Exception('Action took too much time')


def denorm_final(expr, entity_label_map):

    expr = expr.replace('^^http', 'http')
    expr = expr.replace('http', '^^http')
    expr = expr.replace('( ', '(').replace(' )', ')')
            
    entities = re.findall(r'\[ (.*?) \]', expr)
    expr_list = [expr]
    for e in entities:
        # delete the beginning and end space of e
        orig_e = f"[ {e} ]"

        new_expr_list = []
        name_e = e
        if name_e in entity_label_map:
            for expr_i in expr_list:
                for id_map in entity_label_map[name_e]:
                    new_expr_list.append(expr_i.replace(orig_e, id_map))
        
        expr_list = new_expr_list
    expr_list = expr_list[:10]
    return expr_list


def revise_only_name(expr, entity_label_map):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    norm_toks = []
    for t in toks:
        # normalize entity
        if t.startswith('m.') or t.startswith('g.'):
            if t in entity_label_map:
                t = entity_label_map[t]
            else:
                name = get_label(t)
                if name is not None:
                    entity_label_map[t] = name
                    t = name
            t = "[ " + t + " ]"
        # normalize type
        norm_toks.append(t)
    return ' '.join(norm_toks)


def is_valid(answer):
    if answer is not None and answer != '':
        return True
    else:
        return False

def parse_answer(raw_ans_list, original_name=False):
    answers = [ans['text'] if ans['text'] is not None else ans['freebaseId'] for ans in raw_ans_list]
    if not answers:
        answers = ["None"]
    if original_name:
        answers = [get_raw_name(ans) for ans in answers]
    return answers

def parse_answer_id(raw_ans_list):
    answers = [ans['freebaseId'] for ans in raw_ans_list]
    return answers

def compute_f1(pred_set, gold_set):
    pred_set = set(pred_set)
    gold_set = set(gold_set)
    precision = len(pred_set & gold_set) / (len(pred_set) + 1e-8)
    recall = len(pred_set & gold_set) / (len(gold_set) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def compute_hits1_dict(pred_dict, gold_dict):
    score = 0
    for key in gold_dict:
        if key in pred_dict:
            if pred_dict[key][0] in gold_dict[key]:
                score += 1
    return score / len(gold_dict)

# permute the list to perform augmentation
# [A, B, C] -> ["A | B | C", "B | A | C", "C | A | B", ...]
def aug_list_old(org_list, augmentation_num=10):
    all_orders = []
    for i, answer_order in enumerate(itertools.permutations(org_list)):
        all_orders.append(answer_order)
        if i == augmentation_num - 1:
            break
    return [" | ".join(answers_i) for answers_i in all_orders]


def aug_list(org_list):
    augmentation_num = min( max(1, len(org_list) * (len(org_list) - 1)) , 10)
    all_orders = []
    for _ in range(augmentation_num):
        all_orders.append(random.sample(org_list, len(org_list)))
    return [" | ".join(answers_i) for answers_i in all_orders]


# transform a freebase id to a name
def id2name(m_id, id2name_dict):
    if m_id in id2name_dict:
        return id2name_dict[m_id]
    else:
        try:
            name = get_label(m_id)
            if name is not None:
                return name
            else:
                return m_id
        except:
            return m_id


def oracle_rerank(raw_answer_dict, answer_true):
    '''
    choose the answer with the highest F1 score with the true answer
    '''
    new_answer_dict = {
        "SP": [each["answer"] for each in raw_answer_dict["SP"]],
        "QA": [each["answer"] for each in raw_answer_dict["QA"]]
    }
    if len(new_answer_dict["SP"]) == 0:
        if len(new_answer_dict["QA"]) > 0:
            return new_answer_dict["QA"][0], None
        else:
            return ["None"], None
    if len(new_answer_dict["QA"]) == 0:
        return new_answer_dict["SP"][0], raw_answer_dict["SP"][0]["logical_form"]
    # compute the F1 score for top answer
    top_answer_sp = new_answer_dict["SP"][0]
    top_f1_sp = compute_f1(set(top_answer_sp), set(answer_true))
    top_answer_qa = new_answer_dict["QA"][0]
    top_f1_qa = compute_f1(set(top_answer_qa), set(answer_true))
    if top_f1_sp >= top_f1_qa:
        return top_answer_sp, raw_answer_dict["SP"][0]["logical_form"]
    else:
        return top_answer_qa, None



def score_rerank(raw_answer_dict, sp_weight=1.0):
    '''
    SP: [[A, B, C], [B, C]], QA: [[A, B], [B, C]]
    '''
    new_answer_dict = {
        "SP": [each["answer"] for each in raw_answer_dict["SP"]],
        "QA": [each["answer"] for each in raw_answer_dict["QA"]]
    }
    answer2logic = {}
    for each in raw_answer_dict["SP"]:
        if tuple(each["answer"]) not in answer2logic:
            answer2logic[tuple(each["answer"])] = each["logical_form"]
    
    if len(new_answer_dict["QA"]) == 0:
        new_answer_dict["QA"] = [["None"]]

    if len(new_answer_dict["SP"]) == 0:
        if len(new_answer_dict["QA"]) > 0:
            return new_answer_dict["QA"][0], None
        else:
            return ["None"], None
    
    score_dict = {}
    for i in range(len(new_answer_dict["SP"])):
        score_dict["SP{}".format(i)] = 1/(1+i) * sp_weight
        # score_dict["SP{}".format(i)] = (len(new_answer_dict["QA"]) - i) * sp_weight
    for i in range(len(new_answer_dict["QA"])):
        score_dict["QA{}".format(i)] = 1/(1+i) * (1-sp_weight)
        # score_dict["QA{}".format(i)] = (len(new_answer_dict["QA"]) - i) * (1-sp_weight)
    
    answer2score_dict = {}
    
    for i, sp_i in enumerate(new_answer_dict["SP"]):
        if tuple(sp_i) in answer2score_dict:
            continue
        hits = False
        for j, qa_j in enumerate(new_answer_dict["QA"]):
            if set(sp_i) == set(qa_j):
                answer2score_dict[tuple(sp_i)] = score_dict["SP{}".format(i)]
                answer2score_dict[tuple(sp_i)] += score_dict["QA{}".format(j)]
                hits = True
                break
        if not hits:
            answer2score_dict[tuple(sp_i)] = score_dict["SP{}".format(i)]

    for j, qa_j in enumerate(new_answer_dict["QA"]):
        if tuple(qa_j) not in answer2score_dict:
            answer2score_dict[tuple(qa_j)] = score_dict["QA{}".format(j)]    

    # sort the dict based on value
    sorted_score_dict = sorted(answer2score_dict.items(), key=lambda x: x[1], reverse=True)
    top_answer = sorted_score_dict[0][0]

    if top_answer in answer2logic:
        return list(top_answer), answer2logic[top_answer]
    else:
        return list(top_answer), None


def Prefix_to_id_all(in_file_path, name2id_dict, SP_early_break=True):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    new_data = {}
    for key in data:
        task = key.split(":")[-1]
        q_key = ":".join(key.split(":")[:-1])
        if q_key not in new_data:
            new_data[q_key] = {
                "question": data[key]["question"],
                "gold answers": data[key]["gold answers"],
                "predicted answers": {
                    task: data[key]["predicted answers"]
                }
            }
        else:
            new_data[q_key]["predicted answers"][task] = data[key]["predicted answers"]
    print(len(new_data))
    print(list(new_data.keys())[0], new_data[list(new_data.keys())[0]])

    result_dict = {}
    for i, key in tqdm(enumerate(new_data)):
        data_i = new_data[key]
        answers = data_i['predicted answers']
        new_answer_total = {"SP": [], "QA": []}

        SP_answers = answers['SP']
        find_answer = False
        for sp_i, answer in enumerate(SP_answers):
            for result in denorm_final(answer, name2id_dict):
                new_answer = execute_vanilla_s_expr(result)
                if len(new_answer) > 0:
                    new_answer_total["SP"].append({"answer": new_answer, "logical_form": result, "rank": sp_i})
                    find_answer = True
                    break
            if find_answer and SP_early_break:
                break

        QA_answers = answers['QA']
        for qa_i, answer in enumerate(QA_answers):
            if " | " in answer:
                # multiple answers
                total_answers = []
                answer_split = answer.split(" | ")
                for each in answer_split:
                    if each in name2id_dict:
                        total_answers += name2id_dict[each]
                if len(total_answers) > 0:
                    # deduplication but keep original order
                    total_answers = list(OrderedDict.fromkeys(total_answers))
                    new_answer = total_answers
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
            else:
                if answer in name2id_dict:
                    new_answer = name2id_dict[answer]
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
        result_dict[key] = {
                "question": new_data[key]["question"],
                "gold answers": new_data[key]["gold answers"],
                "predicted answers": new_answer_total,
            }
    save_file = in_file_path.replace(".json", "_id_all.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file


def SP_to_id_all(in_file_path, name2id_dict):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    new_data = {}
    for key in data:
        task = "SP"
        q_key = key
        if q_key not in new_data:
            new_data[q_key] = {
                "question": data[key]["question"],
                "gold answers": data[key]["gold answers"],
                "predicted answers": {
                    task: data[key]["predicted answers"]
                }
            }
        else:
            new_data[q_key]["predicted answers"][task] = data[key]["predicted answers"]
    print(len(new_data))
    print(list(new_data.keys())[0], new_data[list(new_data.keys())[0]])

    result_dict = {}
    for i, key in tqdm(enumerate(new_data)):
        data_i = new_data[key]
        answers = data_i['predicted answers']
        new_answer_total = {"SP": [], "QA": []}

        SP_answers = answers['SP']
        find_answer = False
        for sp_i, answer in enumerate(SP_answers):
            for result in denorm_final(answer, name2id_dict):
                new_answer = execute_vanilla_s_expr(result)
                if len(new_answer) > 0:
                    new_answer_total["SP"].append({"answer": new_answer, "logical_form": result, "rank": sp_i})
                    find_answer = True
                    break
            if find_answer:
                break

        result_dict[key] = {
                "question": new_data[key]["question"],
                "gold answers": new_data[key]["gold answers"],
                "predicted answers": new_answer_total,
            }
    save_file = in_file_path.replace(".json", "_id_all.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file


def QA_to_id_all(in_file_path, name2id_dict):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    new_data = {}
    for key in data:
        task = "QA"
        q_key = key
        if q_key not in new_data:
            new_data[q_key] = {
                "question": data[key]["question"],
                "gold answers": data[key]["gold answers"],
                "predicted answers": {
                    task: data[key]["predicted answers"]
                }
            }
        else:
            new_data[q_key]["predicted answers"][task] = data[key]["predicted answers"]
    print(len(new_data))
    print(list(new_data.keys())[0], new_data[list(new_data.keys())[0]])

    result_dict = {}
    for i, key in tqdm(enumerate(new_data)):
        data_i = new_data[key]
        answers = data_i['predicted answers']
        new_answer_total = {"SP": [], "QA": []}

        QA_answers = answers['QA']
        for qa_i, answer in enumerate(QA_answers):
            if " | " in answer:
                # multiple answers
                total_answers = []
                answer_split = answer.split(" | ")
                for each in answer_split:
                    if each in name2id_dict:
                        total_answers += name2id_dict[each]
                if len(total_answers) > 0:
                    # deduplication but keep original order
                    total_answers = list(OrderedDict.fromkeys(total_answers))
                    new_answer = total_answers
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
            else:
                if answer in name2id_dict:
                    new_answer = name2id_dict[answer]
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
        result_dict[key] = {
                "question": new_data[key]["question"],
                "gold answers": new_data[key]["gold answers"],
                "predicted answers": new_answer_total,
            }
    save_file = in_file_path.replace(".json", "_id_all.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file


def QA_to_name_all(in_file_path):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    new_data = {}
    for key in data:
        task = "QA"
        q_key = key
        if q_key not in new_data:
            new_data[q_key] = {
                "question": data[key]["question"],
                "gold answers": data[key]["gold answers"],
                "predicted answers": {
                    task: data[key]["predicted answers"]
                }
            }
        else:
            new_data[q_key]["predicted answers"][task] = data[key]["predicted answers"]
    print(len(new_data))
    print(list(new_data.keys())[0], new_data[list(new_data.keys())[0]])

    result_dict = {}
    for i, key in tqdm(enumerate(new_data)):
        data_i = new_data[key]
        answers = data_i['predicted answers']
        new_answer_total = {"SP": [], "QA": []}

        QA_answers = answers['QA']
        for qa_i, answer in enumerate(QA_answers):
            if " | " in answer:
                # multiple answers
                total_answers = []
                answer_split = answer.split(" | ")
                for each in answer_split:
                    total_answers += [get_raw_name(each)]
                if len(total_answers) > 0:
                    # deduplication but keep original order
                    total_answers = list(OrderedDict.fromkeys(total_answers))
                    new_answer = total_answers
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
            else:
                new_answer = [get_raw_name(answer)]
                new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
        result_dict[key] = {
                "question": new_data[key]["question"],
                "gold answers": new_data[key]["gold answers"],
                "predicted answers": new_answer_total,
            }
    save_file = in_file_path.replace(".json", "_name_all.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file


def Combine_QA_SP(in_file_path_qa, in_file_path_sp):
    with open(in_file_path_qa, "r") as f:
        data_qa = json.load(f)
    print(len(data_qa))
    print(list(data_qa.keys())[0], data_qa[list(data_qa.keys())[0]])
    with open(in_file_path_sp, "r") as f:
        data_sp = json.load(f)
    print(len(data_sp))
    print(list(data_sp.keys())[0], data_sp[list(data_sp.keys())[0]])
    assert len(data_sp) == len(data_qa)
    new_data = {}
    for key in data_qa:
        new_data[key + ":QA"] = data_qa[key]
    for key in data_sp:
        new_data[key + ":SP"] = data_sp[key]
    print(len(new_data))
    save_file = in_file_path_qa.replace(".json", "_combine.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(new_data, wf, indent=2)
    return save_file



def Prefix_to_name_all(in_file_path, name2id_dict, id2name_dict):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    new_data = {}
    for key in data:
        task = key.split(":")[-1]
        q_key = ":".join(key.split(":")[:-1])
        if q_key not in new_data:
            new_data[q_key] = {
                "question": data[key]["question"],
                "gold answers": data[key]["gold answers"],
                "predicted answers": {
                    task: data[key]["predicted answers"]
                }
            }
        else:
            new_data[q_key]["predicted answers"][task] = data[key]["predicted answers"]
    print(len(new_data))
    print(list(new_data.keys())[0], new_data[list(new_data.keys())[0]])

    result_dict = {}
    for i, key in tqdm(enumerate(new_data)):
        data_i = new_data[key]
        answers = data_i['predicted answers']
        new_answer_total = {"SP": [], "QA": []}

        SP_answers = answers['SP']
        find_answer = False
        for sp_i, answer in enumerate(SP_answers):
            for result in denorm_final(answer, name2id_dict):
                new_answer = execute_vanilla_s_expr(result)
                if len(new_answer) > 0:
                    new_answer = [id2name(answer_i, id2name_dict) for answer_i in new_answer]
                    new_answer_total["SP"].append({"answer": new_answer, "logical_form": result, "rank": sp_i})
                    find_answer = True
                    break
            if find_answer:
                break

        QA_answers = answers['QA']
        for qa_i, answer in enumerate(QA_answers):
            if " | " in answer:
                # multiple answers
                total_answers = []
                answer_split = answer.split(" | ")
                for each in answer_split:
                    # if each in name2id_dict:
                    total_answers += [get_raw_name(each)]
                if len(total_answers) > 0:
                    # deduplication but keep original order
                    total_answers = list(OrderedDict.fromkeys(total_answers))
                    new_answer = total_answers
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
            else:
                # if answer in name2id_dict:
                new_answer = [get_raw_name(answer)]
                new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
        result_dict[key] = {
                "question": new_data[key]["question"],
                "gold answers": new_data[key]["gold answers"],
                "predicted answers": new_answer_total,
            }
    save_file = in_file_path.replace(".json", "_name_all.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file


def Concat_to_id_all(in_file_path, name2id_dict):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    new_data = {}
    for key in data:
        sp_answers, qa_answers = [], []
        for each in data[key]["predicted answers"]:
            sp_answers.append(each.split("expression: ")[-1].split(" answer: ")[0])
            qa_answers.append(each.split("answer: ")[-1])
        new_data[key] = {
            "question": data[key]["question"],
            "gold answers": data[key]["gold answers"],
            "predicted answers": {
                "SP": sp_answers,
                "QA": qa_answers
            }
        }
    print(len(new_data))
    print(list(new_data.keys())[0], new_data[list(new_data.keys())[0]])

    result_dict = {}
    for i, key in tqdm(enumerate(new_data)):
        data_i = new_data[key]
        answers = data_i['predicted answers']
        new_answer_total = {"SP": [], "QA": []}

        SP_answers = answers['SP']
        for sp_i, answer in enumerate(SP_answers):
            for result in denorm_final(answer, name2id_dict):
                new_answer = execute_vanilla_s_expr(result)
                if len(new_answer) > 0:
                    new_answer_total["SP"].append({"answer": new_answer, "logical_form": result, "rank": sp_i})
                    break

        QA_answers = answers['QA']
        for qa_i, answer in enumerate(QA_answers):
            if " | " in answer:
                # multiple answers
                total_answers = []
                answer_split = answer.split(" | ")
                for each in answer_split:
                    if each in name2id_dict:
                        total_answers += name2id_dict[each]
                if len(total_answers) > 0:
                    # deduplication but keep original order
                    total_answers = list(OrderedDict.fromkeys(total_answers))
                    new_answer = total_answers
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
            else:
                if answer in name2id_dict:
                    new_answer = name2id_dict[answer]
                    new_answer_total["QA"].append({"answer": new_answer, "rank": qa_i})
        result_dict[key] = {
                "question": new_data[key]["question"],
                "gold answers": new_data[key]["gold answers"],
                "predicted answers": new_answer_total,
            }
    save_file = in_file_path.replace(".json", "_id_all.json")
    print(save_file)
    with open(save_file, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file


def sort_SP_with_QA(SP_answers, QA_answers):
    # assign scores to QA answers based on rank
    QA_answers_score_dict = defaultdict(float)
    for qa_i, answer in enumerate(QA_answers):
        QA_answers_score_dict[answer] = 1/(1+qa_i)
    # sort SP answers based on QA answers
    SP_answers_sorted = []
    for sp_i, answer in enumerate(SP_answers):
        SP_answers_sorted.append({"answer": answer, "score": QA_answers_score_dict[answer]})
    SP_answers_sorted = sorted(SP_answers_sorted, key=lambda x: x["score"], reverse=True)
    SP_answers_sorted_answers = [each["answer"] for each in SP_answers_sorted]
    return SP_answers_sorted_answers


def answer_ensemble(in_file_path, mode="SP_first", sp_weight=1.0, gold_answers=None):
    '''
    combine the SP and QA answer to the final answers
    '''
    def get_top_k(answer_list, k=1):
        top_k = min(k, len(answer_list))
        output_answer = []
        for i in range(top_k):
            output_answer += answer_list[i]["answer"]
        output_answer = list(OrderedDict.fromkeys(output_answer))
        return output_answer
    
    assert "_all.json" in in_file_path
    with open(in_file_path, "r") as f:
        data = json.load(f)
    # print(len(data))
    # print(list(data.keys())[0], data[list(data.keys())[0]])

    num_sp = 0
    for key in data:
        pred_answers = data[key]["predicted answers"]
        if gold_answers is not None:
            answer_true = gold_answers[key]
        if mode == "SP_first":
            data[key]["predicted answers"] = []
            QA_answers = []
            if len(pred_answers["QA"]) > 0:
                # data[key]["predicted answers"] = pred_answers["QA"][0]["answer"]
                data[key]["predicted answers"] = get_top_k(pred_answers["QA"], k=1)
                QA_answers = pred_answers["QA"][0]["answer"]
                # QA_answers = [pred_answers["QA"][i]["answer"][0] for i in range(len(pred_answers["QA"]))]
            if len(pred_answers["SP"]) > 0:
                # data[key]["predicted answers"] = pred_answers["SP"][0]["answer"]
                data[key]["predicted answers"] = sort_SP_with_QA(pred_answers["SP"][0]["answer"], QA_answers)
                data[key]["logical_form"] = pred_answers["SP"][0]["logical_form"]
                num_sp += 1
        elif mode == "SP_only":
            data[key]["predicted answers"] = []
            if len(pred_answers["SP"]) > 0:
                data[key]["predicted answers"] = pred_answers["SP"][0]["answer"]
                data[key]["logical_form"] = pred_answers["SP"][0]["logical_form"]
                num_sp += 1
        elif mode == "QA_only":
            data[key]["predicted answers"] = []
            if len(pred_answers["QA"]) > 0:
                # data[key]["predicted answers"] = pred_answers["QA"][0]["answer"]
                # combine top-k QA answers
                data[key]["predicted answers"] = get_top_k(pred_answers["QA"], k=1)
        elif mode == "Score":
            data[key]["predicted answers"], logical_form = score_rerank(pred_answers, sp_weight)
            if len(pred_answers["QA"]) > 0:
                data[key]["predicted answers"] = sort_SP_with_QA(data[key]["predicted answers"], pred_answers["QA"][0]["answer"])
            if logical_form is not None:
                num_sp += 1
                data[key]["logical_form"] = logical_form
        elif mode == "oracle":
            data[key]["predicted answers"], logical_form = oracle_rerank(pred_answers, answer_true)
            if len(pred_answers["QA"]) > 0:
                data[key]["predicted answers"] = sort_SP_with_QA(data[key]["predicted answers"], pred_answers["QA"][0]["answer"])
            if logical_form is not None:
                num_sp += 1
                data[key]["logical_form"] = logical_form
    
    # print("num_sp:", num_sp)
    save_file = in_file_path.replace("_all.json", "_final.json")
    # print(save_file)
    with open(save_file, "w") as wf:
        json.dump(data, wf, indent=2)
    return save_file
    

def Joint_to_id(in_file_path, name2id_dict):

    with open(in_file_path, "r") as f:
        data = json.load(f)
    print(len(data))
    print(list(data.keys())[0], data[list(data.keys())[0]])

    result_dict = {}
    num_qa = 0
    for i, key in tqdm(enumerate(data)):
        data_i = data[key]
        answers = data_i['predicted answers']
        new_answer = []
        find_answer = False
        logical_form = "None"
        for answer in answers:
            if "expression:" in answer:
                answer = answer.split("expression: ")[-1]
                for result in denorm_final(answer, name2id_dict):
                    new_answer = execute_vanilla_s_expr(result)
                    if len(new_answer) > 0:
                        find_answer = True
                        logical_form = result
                        break
                if find_answer:
                    break
            elif "answer:" in answer:
                answer = answer.split("answer: ")[-1]
                num_qa += 1
                if " | " in answer:
                    # multiple answers
                    total_answers = []
                    answer_split = answer.split(" | ")
                    for each in answer_split:
                        if each in name2id_dict:
                            total_answers += name2id_dict[each]
                    if len(total_answers) > 0:
                        # deduplication but keep original order
                        total_answers = list(OrderedDict.fromkeys(total_answers))
                        new_answer = total_answers
                        break
                else:
                    if answer in name2id_dict:
                        new_answer = name2id_dict[answer]
                        break
        result_dict[key] = {
                "question": data[key]["question"],
                "gold answers": data[key]["gold answers"],
                "predicted answers": [new_answer],
                "logical_form": logical_form,
            }
    print(num_qa)
    save_file_id = in_file_path.replace(".json", "_id.json")
    print(save_file_id)
    with open(save_file_id, "w") as wf:
        json.dump(result_dict, wf, indent=2)
    return save_file_id


def execute_vanilla_s_expr(s_expr):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10) #Set the parameter to the amount of seconds you want to wait
    try:
        # print('parse', query_expr)
        sparql_query = lisp_to_sparql(s_expr)
        # print('sparql', sparql_query)
        # set maximum time as 100 seconds
        denotation = execute_query(sparql_query)
    except:
        denotation = []
    signal.alarm(10) #Resets the alarm to 10 new seconds
    signal.alarm(0) #Disables the alarm
    return denotation


def denormalize(expr, entity_label_map):

    expr = expr.replace('^^http', 'http')
    expr = expr.replace('http', '^^http')
    expr = expr.replace('( ', '(').replace(' )', ')')
            
    entities = re.findall(r'\[ (.*?) \]', expr)
    expr_list = [expr]
    for e in entities:
        # delete the beginning and end space of e
        orig_e = f"[ {e} ]"

        new_expr_list = []
        name_e = e
        if name_e in entity_label_map:
            for expr_i in expr_list:
                for id_map in entity_label_map[name_e]:
                    new_expr_list.append(expr_i.replace(orig_e, id_map))
        
        expr_list = new_expr_list
    expr_list = expr_list[:10]
    return expr_list


def SP_to_id(in_file, name2id_dict):
    with open(in_file, 'r') as rf:
        data = json.load(rf)
    print(len(data))
    print("Data Example:")
    print(data[list(data.keys())[0]])
    new_data = {}
    wrong_data = []
    num_no_answers = 0
    for key in tqdm(data):
        logical_form = "None"
        answers = []
        for predict_exp in data[key]["predicted answers"]:
            predict_exp = denormalize(predict_exp, name2id_dict)
            if isinstance(predict_exp, list):
                for predict_exp_i in predict_exp:
                    predict_answers = execute_vanilla_s_expr(predict_exp_i)
                    if len(predict_answers) != 0:
                        logical_form = predict_exp_i
                        answers = predict_answers
                        break
            else:
                predict_answers = execute_vanilla_s_expr(predict_exp)
                if len(predict_answers) != 0:
                    logical_form = predict_exp
                    answers = predict_answers
            if len(answers) != 0:
                break
        
        if len(answers) == 0:
            num_no_answers += 1
            data[key].update({"id": key})
            wrong_data.append(data[key])
        
        new_data[key] = {
            "question": data[key]["question"],
            "gold answers": data[key]["gold answers"],
            "logical_form": logical_form,
            "predicted answers": [answers]
        }
    print("num_no_answers: ", num_no_answers)
    print("Output Example:")
    print(new_data[list(new_data.keys())[0]])
    out_file = in_file.replace(".json", "_id.json")
    with open(out_file, 'w') as wf:
        json.dump(new_data, wf, indent=2)
    wrong_file = in_file.replace(".json", "_wrong.json")
    with open(wrong_file, "w") as wf:
        json.dump(wrong_data, wf, indent=2)
    return out_file


def id_to_name(in_file_path, id2name_dict):
    with open(in_file_path, "r") as rf:
        data = json.load(rf)
    # print("Data example: ")
    # print(data[list(data.keys())[0]])
    new_data = {}
    for key in tqdm(data):
        new_answer_list = []
        for answer in data[key]["predicted answers"]:
            new_answer_list.append(id2name(answer, id2name_dict))
        new_data[key] = {
            "question": data[key]["question"],
            "gold answers": data[key]["gold answers"],
            "predicted answers": new_answer_list
        }
    # print("Output example: ")
    # print(new_data[list(new_data.keys())[0]])
    out_save_path = in_file_path.replace(".json", "_name.json")
    # print("Output File:", os.path.basename(out_save_path))
    with open(out_save_path, "w") as wf:
        json.dump(new_data, wf, indent=2)
    return out_save_path

######## Transform results to official evaluation format ########

def to_webqsp_format(in_file_path):
    with open(in_file_path, "r") as rf:
        data = json.load(rf)
    new_data = []
    for key in data:
        new_data.append({"QuestionId": key, "Answers": data[key]["predicted answers"]})
    # print("Output example: ")
    # print(new_data[0])
    out_save_path = in_file_path.replace(".json", "_eval.json")
    # print("Output File:", os.path.basename(out_save_path))
    with open(out_save_path, "w") as wf:
        json.dump(new_data, wf, indent=2)
    return out_save_path


def to_cwq_format(in_file_path):
    with open(in_file_path, "r") as rf:
        data = json.load(rf)
    new_data = []
    for key in data:
        answer = data[key]["predicted answers"]
        if len(answer) > 0:
            # randomly pick one as answer
            answer = answer[0]
            # answer = random.sample(answer, 1)[0]
        else:
            answer = "None"
        new_data.append({"ID": key, "answer": answer.lower().strip()})
    # print("Output example: ")
    # print(new_data[0])
    out_save_path = in_file_path.replace(".json", "_eval.json")
    # print("Output File:", os.path.basename(out_save_path))
    with open(out_save_path, "w") as wf:
        json.dump(new_data, wf, indent=2)
    return out_save_path


def to_grailqa_format(in_file_path):
    with open(in_file_path, "r") as rf:
        data = json.load(rf)
    new_data = []
    for key in data:
        new_data_i = {
            "qid": key,
            "logical_form": data[key]["logical_form"] if "logical_form" in data[key] else "None",
            "answer": data[key]["predicted answers"]
        }
        new_data.append(new_data_i)
    # print("Output example: ")
    # print(new_data[0])
    out_save_path = in_file_path.replace(".json", "_eval.json")
    # print("Output File:", os.path.basename(out_save_path))
    lines = [json.dumps(x) for x in new_data]
    with open(out_save_path, 'w') as wf:
        wf.writelines([x+'\n' for x in lines])
    return out_save_path