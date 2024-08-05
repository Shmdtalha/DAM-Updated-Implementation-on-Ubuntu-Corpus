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

def build_response_dict(filepath):
	response_dict = {}
	with open(filepath, 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			if len(row) < 2:
				continue
			response_id = int(row[0])
			response_dict[response_id] = row[1]  # Context used as response
	return response_dict

def tokenize(vocab, utt):
	return [vocab.get(tok, vocab["UNKNOWN"]) for tok in utt.split()]

def process_file(file_path, vocab, response_dict, enforce=0):
	y = []
	c = []  # token ids for each context
	r = []  # token ids for each response
	fbase = file_path.split("/")[-1]
	skipCount = 0
	with open(file_path, 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		lineno = 0
		for row in reader:
			lineno += 1
			utt = row[1]
			valids = row[2]
			invalids = row[3]
			isValid = valids != "NA"
			splits = [[], []]
			if valids != "NA":
				splits[0] = [int(x) for x in valids.split("|")]
			if invalids != "NA":
				splits[1] = [int(x) for x in invalids.split("|")]
			respCount = len(splits[0]) + len(splits[1])
			if enforce != 0 and respCount != enforce:
				#print("{} line: {} skipped, responseCount {} != enforce {}".format(
				#	fbase, lineno, respCount, enforce))
				skipCount += 1
				continue
			for label in xrange(len(splits)):
				for id in splits[label]:
					y.append((label + 1) % 2)
					c.append(tokenize(vocab, utt))
					r.append(tokenize(vocab, response_dict[id]))
			# not appending EOS to end of contexts because __eot__ is there at
			# end of each utterance in V2 dataset
	if enforce != 0:
		print("{} Records skipped due to enforce={} is = {}".format(
			fbase, enforce, skipCount
			))
	return y, c, r

def create_data_pkl(vocab_file, train_file, test_file, valid_file, output_file, response_file):
	vocab = load_vocab(vocab_file)
	EOS = vocab['eot']
	files = [train_file, test_file, valid_file]

	response_dict = build_response_dict(response_file)
	train_y, train_c, train_r = process_file(train_file, vocab, response_dict)
	test_y, test_c, test_r = process_file(test_file, vocab, response_dict, 10)
	valid_y, valid_c, valid_r = process_file(valid_file, vocab, response_dict, 10)

	assert len(train_y) == len(train_c) == len(train_r)
	assert len(test_y) == len(test_c) == len(test_r)
	assert len(valid_y) == len(valid_c) == len(valid_r)

	data = ({
		'y': train_y, 'c': train_c, 'r': train_r
	}, {
		'y': valid_y, 'c': valid_c, 'r': valid_r
	}, {
		'y': test_y, 'c': test_c, 'r': test_r
	})

	with open(output_file, 'wb') as f:
		pickle.dump(data, f)

#Modify these paths depending on where your data has been stored
vocab_file = 'data/ubuntu/vocab.txt'
train_file = 'data/ubuntu/train.txt'
test_file = 'data/ubuntu/test.txt'
valid_file = 'data/ubuntu/valid.txt'
output_file = 'data/ubuntu/data.pkl'
response_file = 'data/ubuntu/responses.txt'

create_data_pkl(vocab_file, train_file, test_file, valid_file, output_file, response_file)
