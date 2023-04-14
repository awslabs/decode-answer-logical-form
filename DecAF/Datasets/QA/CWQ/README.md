## ComplexWebQuestions

Dataset official website: https://www.tau-nlp.sites.tau.ac.il/compwebq

1. Download Dataset and extract it to `${DATA_DIR}/tasks/QA/CWQ/raw/` folder. The original test data does not contain the answer, so we use the [ComplexWebQuestions_test_wans.json](https://drive.google.com/file/d/1NiJsErTKVmn3NW9vjVPMWwhBgR92S7Cd/view?usp=share_link) from [KBQA-GST](https://github.com/lanyunshi/KBQA-GST). 
The directory structure should be like this:
```
${DATA_DIR}/tasks/QA/CWQ/raw
├── ComplexWebQuestions_train.json
├── ComplexWebQuestions_dev.json
├── ComplexWebQuestions_test.json # replaced by ComplexWebQuestions_test_wans.json
├── README.txt
```

2. Process QA pairs:
```
python preprocess_QA.py --input_dir ${DATA_DIR}/tasks/QA/CWQ/raw --output_dir ${DATA_DIR}/tasks/QA/CWQ
```

3. Disambiguate entities:
```
python ../disambiguate.py --data_dir ${DATA_DIR}/tasks/QA/CWQ
```

4. Process SP pairs with s-expression:
```
python parse_sparql.py --input_dir ${DATA_DIR}/tasks/QA/CWQ/raw
```
The parse_sparql.py is modified based on the original [parse_sparql.py](https://github.com/salesforce/rng-kbqa/blob/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/parse_sparql.py) from [RnG-KBQA](https://github.com/salesforce/rng-kbqa).

5. Finally, combine the QA pairs and SP pairs:
```
python preprocess_SP.py --data_dir ${DATA_DIR}/tasks/QA/CWQ
```

6. For later evaluation, download the official [eval_script.py](https://github.com/alontalmor/WebAsKB/blob/master/eval_script.py) and put it in the same folder as this README.md.