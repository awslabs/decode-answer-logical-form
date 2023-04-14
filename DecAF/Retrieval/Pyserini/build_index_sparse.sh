#!/bin/sh

# Build the index for the general knowledge base using pyserini.

Freebase="${DATA_DIR}/knowledge_source/Freebase/processed"

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${Freebase}/document \
  --index ${Freebase}/index/pyserini_bm25 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 10 \
  --storePositions --storeDocvectors --storeRaw
