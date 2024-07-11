# DAM-Updated Implementation on Ubuntu Corpus
This repository contains the updated implementation of Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network (DAM)
This model is compaible with python 2.7.
## Download Preprocessing Files
Download the files below if you do not wish to preprocess the data. \
[word_embedding.pkl](https://drive.google.com/file/d/1jSBQsdF5Awk8IDKpJjUtJHR83RJfWYBj/view?usp=sharing) \
[data.pkl](https://drive.google.com/file/d/1-9BXvk7aCDS70G1MfWH6aRe0cXONW1vh/view?usp=sharing) 

Please note that these files may not work with older versions of Python. If you wish to work in a different version or dataset, preprocess the data with the steps below.
## Dataset
If you wish to preprocess the data, you may download the dataset here: [Ubuntu Dialogue Corpus](https://drive.google.com/drive/folders/1cm1v3njWPxG5-XhEUpGH25TMncaPR7OM?usp=sharing) 

## Preprocessing
1. Create word_embedding.pkl \
You might have to modify your file paths in `word_embedding.py`. This is only for the files in the dataset.
```python
!python word_embedding.py
```
This will create a file called word_embedding.pkl. Place this in the data directory.
2. Create data.pkl \
You might have to modify your file paths in `data_.py`. This is only for the files in the dataset.
```python
!python data_.py
```
This will create a file called data.pkl.The path should be /data/ubuntu/data.pkl
## Training
Training will begin after running the shell file
```python
!sh run.sh
```
## Evaluation
To evaluate the model, you must first set the `init_model` flag in main.py. After this, you may run the shell file.
```python
!sh run.sh
```
## Acknowledgments
[DAM GitHub Repository](https://github.com/baidu/Dialogue/tree/master/DAM) \
[Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://aclanthology.org/P18-1103.pdf)
