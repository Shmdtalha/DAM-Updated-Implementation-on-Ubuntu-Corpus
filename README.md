# DAM-Updated-Implementation-on-Ubuntu-Corpus
This repository contains the updated implementation of Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network (DAM)
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
Training will begin after running the line below
```python
sh run.sh
```
## Evaluation
To evaluate the model, run this
```python
!python /utils/evaluation.py
```
