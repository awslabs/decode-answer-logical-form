# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
from tqdm import tqdm
from collections import defaultdict

class Relation:
    def __init__(self, line):
        if line is None:
            self.subj = self.rel = self.obj = None
            return
        e1, rel, e2 = line.strip().split("\t")
        self.subj = e1
        self.rel = rel
        self.obj = e2
    
    def __hash__(self):
        return hash((self.subj, self.rel, self.obj))
        
    def _filter_relation(self):
        relation = self.rel
        if relation == "type.object.name":
            return True
        return False

    def should_ignore(self, id2name_dict):
        if self._filter_relation():
            return True
        return False
    
    def __repr__(self):
        return f"Subj: {self.subj}; Rel: {self.rel}; Obj: {self.obj}"


def convert_relation_to_text(relation, entity_names):
    if isinstance(relation, Relation):
        subj, rel, obj = relation.subj, relation.rel, relation.obj
    else:
        subj, rel, obj = relation

    # subject
    if subj in entity_names:
        subj_surface = entity_names[subj]
    else:
        subj_surface = subj
        
    # object
    if obj in entity_names:
        obj_surface = entity_names[obj]
    else:
        obj_surface = obj
            
    # relation
    # e.g. film.film.other_crew
    # replace '.' and '_' with ' '
    rel_surface = rel.replace('.', ' ')
    rel_surface = rel_surface.replace('_', ' ')
    
    return ' '.join([subj_surface, rel_surface, obj_surface, '.'])


# replace "{name} v2" to "{name}"
def get_raw_name(name_wversion):
    dict_name = name_wversion.split(" ")
    if dict_name[-1].startswith("v") and dict_name[-1][1:].isnumeric():
        dict_name = " ".join(dict_name[:-1])
    else:
        dict_name = " ".join(dict_name)
    return dict_name


def load_nameid_dict(file_dir, lower):
    print("Loading name2id and id2name dict...")
    name2id_dict = defaultdict(list)
    id2name_dict = {}
    for file in tqdm(os.listdir(file_dir)):
        with open(os.path.join(file_dir, file), 'r') as rf:
            data_input = csv.reader(rf, delimiter="\t")
            for row in data_input:
                if lower:
                    procesed_name = row[2].lower()
                else:
                    procesed_name = row[2]
                name2id_dict[procesed_name].append(row[0])
                id2name_dict[row[0]] = procesed_name
    return name2id_dict, id2name_dict