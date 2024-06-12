import csv
import numpy as np
import pickle

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r') as file:
        for line in file:
            word, index = line.strip().split('\t')
            vocab[word] = int(index)
    return vocab

def build_response_dict(files):
    response_dict = {}
    for file_path in files:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if len(row) > 3:  # ID + response
                    response_id = row[3]
                    if response_id != 'NA':
                        response_dict[response_id] = row[1]  # Context used as response
    return response_dict

def process_file(file_path, vocab, response_dict, EOS):
    y = []
    c = []  # token ids for each context
    r = []  # token ids for each response
    missing_responses = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            label = int(row[0])
            context = row[1]
            response_id = row[3]
            context_ids = [vocab.get(token, vocab['UNKNOWN']) for token in context.split()]
            context_ids.append(EOS)  # Append EOS at the end of each context
            c.append(context_ids)  # Store as a list
            y.append(label)
            # Fetch and process the response using ID
            if response_id in response_dict:
                response_text = response_dict[response_id]
                response_ids = [vocab.get(token, vocab['UNKNOWN']) for token in response_text.split()]
                r.append(response_ids)  # Append the list of response ids
            else:
                missing_responses += 1
                r.append([vocab['UNKNOWN']])  # placeholder for missing responses
    print(f'Missing responses: {missing_responses}')
    return np.array(y), np.array(c, dtype=object), np.array(r, dtype=object)

def create_data_pkl(vocab_file, train_file, test_file, valid_file, output_file):
    vocab = load_vocab(vocab_file)
    EOS = vocab['eot']
    files = [train_file, test_file, valid_file]

    response_dict = build_response_dict(files)
    train_y, train_c, train_r = process_file(train_file, vocab, response_dict, EOS)
    test_y, test_c, test_r = process_file(test_file, vocab, response_dict, EOS)
    valid_y, valid_c, valid_r = process_file(valid_file, vocab, response_dict, EOS)

    assert len(train_y) == len(train_c) == len(train_r)
    assert len(test_y) == len(test_c) == len(test_r)
    assert len(valid_y) == len(valid_c) == len(valid_r)

    data = ({
        'y': train_y, 'c': train_c, 'r': train_r
    }, {
        'y': test_y, 'c': test_c, 'r': test_r
    }, {
        'y': valid_y, 'c': valid_c, 'r': valid_r
    })

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

#Modify these paths depending on where your data has been stored
vocab_file = 'data/ubuntu/vocab.txt'
train_file = 'data/ubuntu/train.txt'
test_file = 'data/ubuntu/test.txt'
valid_file = 'data/ubuntu/valid.txt'
output_file = 'data/ubuntu/data.pkl'

create_data_pkl(vocab_file, train_file, test_file, valid_file, output_file)
