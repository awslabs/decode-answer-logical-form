# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

INDEX_DIR = os.environ["DATA_DIR"] + "/knowledge_source"
INDEX_MAP_DICT = {
    "Freebase": f"{INDEX_DIR}/Freebase/processed/index/pyserini_bm25",
}