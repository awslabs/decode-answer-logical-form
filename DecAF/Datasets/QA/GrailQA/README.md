## GrailQA

Dataset official website: https://dki-lab.github.io/GrailQA/

1. Download Dataset and extract it to `${DATA_DIR}/tasks/QA/GrailQA/raw/` folder. 
The directory structure should be like this:
```
${DATA_DIR}/tasks/QA/GrailQA/raw
├── grailqa_v1.0_train.json
├── grailqa_v1.0_test_public.json
├── grailqa_v1.0_dev.json
├── README.txt
```

2. Process QA pairs:
```
python preprocess_QA.py --input_dir ${DATA_DIR}/tasks/QA/GrailQA/raw --output_dir ${DATA_DIR}/tasks/QA/GrailQA
```

3. Disambiguate entities:
```
python ../disambiguate.py --data_dir ${DATA_DIR}/tasks/QA/GrailQA 
```

4. Finally, combine the QA pairs and SP pairs:
```
python preprocess_SP.py --data_dir ${DATA_DIR}/tasks/QA/GrailQA
```

5. For later evaluation, download the official [evaluate.py](https://worksheets.codalab.org/bundles/0x2d13989c17e44690ab62cc4edc0b900d/) and put it in the same folder as this README.md.
