# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
evaluation functions for retrieval
'''

import unicodedata
from collections import defaultdict
import numpy as np
import regex as re
from tqdm import tqdm
from DecAF.Datasets.QA.utils import parse_answer


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        if tokenizer is None:
            text = text.lower()
            for single_answer in answers:
                norm_answer = _normalize(single_answer).lower()
                if norm_answer in text:
                    return True
        else:
            text = tokenizer.tokenize(text).words(uncased=True)

            for single_answer in answers:
                single_answer = _normalize(single_answer)
                single_answer = tokenizer.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)

                for i in range(0, len(text) - len(single_answer) + 1):
                    if single_answer == text[i : i + len(single_answer)]:
                        return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False

def _normalize(text):
    return unicodedata.normalize("NFD", text)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None

def eval_top_k_one(data_i, top_k=100, tokenizer=None):
    recall = 0
    answers = parse_answer(data_i['Answers'], original_name=True)
    for answer in answers:
        for ctx in data_i['ctxs'][:top_k]:
            context = ctx['title'] + " " + ctx['text']
            if has_answer([answer], context, tokenizer, "string"):
                recall += 1
                break
    return recall / (len(answers) + 1e-8)

def recall_ctx(ctx, answers, tokenizer=None):
    context = ctx['title'] + " " + ctx['text']
    recall = 0
    for answer in answers:
        if has_answer([answer], context, tokenizer, "string"):
            recall += 1
    return recall / (len(answers) + 1e-8)

def eval_top_k(output_data, top_k_list=[1, 20, 50, 100, 200, 500], tokenizer=None):
    print("Evaluation")
    hits_dict = defaultdict(int)
    recall_dict = defaultdict(float)
    num_tokens_dict = defaultdict(list)
    top_k_list = [k for k in top_k_list if k <= len(output_data[0]['ctxs'])]
    for data_i in tqdm(output_data):
        for k in top_k_list:
            recall = eval_top_k_one(data_i, top_k=k, tokenizer=tokenizer)
            if recall > 0:
                hits_dict[k] += 1
            recall_dict[k] += recall
            num_tokens_dict[k].append(sum([len(ctx["text"].split(" "))+len(ctx["title"].split(" ")) for ctx in data_i['ctxs'][:k]]))
    for k in top_k_list:
        print("Top {}".format(k), 
              "Hits: ", round(hits_dict[k] * 100 / len(output_data), 1), 
              "Recall: ", round(recall_dict[k] * 100 / len(output_data), 1))
