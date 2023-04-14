## WebQSP

Dataset official website: https://www.microsoft.com/en-us/download/details.aspx?id=52763

1. Download Dataset and extract it to `${DATA_DIR}/tasks/QA/WebQSP/raw`.
The directory structure should be like this:
```
${DATA_DIR}/tasks/QA/WebQSP/raw
├── data
├── doc
├── eval
├── ReadMe.txt
```

2. Process QA pairs:
```
python preprocess_QA.py --input_dir ${DATA_DIR}/tasks/QA/WebQSP/raw/data --output_dir ${DATA_DIR}/tasks/QA/WebQSP
```

3. Disambiguate entitiy names in QA pairs:
```
python ../disambiguate.py --data_dir ${DATA_DIR}/tasks/QA/WebQSP
```

4. Process SP pairs with s-expression, download the [parse_sparql.py](https://github.com/salesforce/rng-kbqa/blob/2b6ef28e7724f11181f59589398894a1d0617455/WebQSP/parse_sparql.py) from [RnG-KBQA](https://github.com/salesforce/rng-kbqa).
```
python parse_sparql.py # you need to change data path in the code
```

5. Finally, combine the QA pairs and SP pairs:
```
python preprocess_SP.py --data_dir ${DATA_DIR}/tasks/QA/WebQSP
```


