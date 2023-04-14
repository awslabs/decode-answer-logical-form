## FreebaseQA

Dataset official website: https://github.com/kelvin-jiang/FreebaseQA

1. Download the dataset and extract it to `${DATA_DIR}/tasks/QA/FreebaseQA/raw/` folder.
The directory structure should be like this:
```
${DATA_DIR}/tasks/QA/FreebaseQA/raw
├── FreebaseQA-train.json
├── FreebaseQA-dev.json
├── FreebaseQA-eval.json
├── FreebaseQA-partial.json
```

2. Process QA pairs:
```
python preprocess_QA.py --input_dir ${DATA_DIR}/tasks/QA/FreebaseQA/raw --output_dir ${DATA_DIR}/tasks/QA/FreebaseQA
```

3. Disambiguate entities:
```
python ../disambiguate.py --data_dir ${DATA_DIR}/tasks/QA/FreebaseQA
```
