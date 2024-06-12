import numpy as np
import pickle
from gensim.models import Word2Vec
import os

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, index = line.strip().split('\t')
            vocab[word] = int(index)
    return vocab

class SentenceIterator:
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        files = ['train.txt', 'valid.txt', 'test.txt']
        for file in files:
            path = os.path.join(self.dirname, file)
            with open(path, encoding='utf-8') as f:
                for line in f:
                    dialogue = line.strip().split('\t')[1]
                    yield dialogue.split()

def train_embeddings(data_dir):
    sentences = SentenceIterator(data_dir)
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=4)
    return model

#Modify paths here if necessary
vocab_file = 'data/ubuntu/vocab.txt'
vocab = load_vocab(vocab_file)

data_dir = 'data/ubuntu'
word2vec_model = train_embeddings(data_dir)

embedding_matrix = np.zeros((len(vocab) + 1, 200))
for word, idx in vocab.items():
    if word in word2vec_model.wv:
        embedding_matrix[idx] = word2vec_model.wv[word]

with open('./data/word_embedding.pkl', 'wb') as f:
    pickle.dump(embedding_matrix, f)
