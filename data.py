import csv
import numpy as np
import pickle

vocab_file = 'data/ubuntu/vocab.txt'
train_file = 'data/ubuntu/train.txt'
test_file = 'data/ubuntu/test.txt'
valid_file = 'data/ubuntu/valid.txt'
output_file = 'data/ubuntu/data.pkl'
response_file = 'data/ubuntu/responses.txt'

eos = None
eot = None
eou = None
vocab = {}

responses = {}

def load_vocab(vocab_file):
	global eos, eot, eou, vocab
	vocab = {}
	eos = 0
	with open(vocab_file, 'r') as file:
		for line in file:
			word, index = line.strip().split('\t')
			vocab[word] = int(index)
			eos = max(vocab[word], eos)
	eos += 1
	eot = vocab["__eot__"]
	eou = vocab["__eou__"]

def build_responses(filepath):
	global responses
	response = {}
	with open(filepath, 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			if len(row) >= 2:
				responses[int(row[0])] = row[1]  # Context used as response

def tokenize(utt):
	global eos, eot, eou, vocab
	ret = []
	arr = [vocab.get(tok, vocab["UNKNOWN"]) for tok in utt.split()]
	i = 0
	while i < len(arr):
		if arr[i] != eot:
			ret.append(arr[i])
		i += 1
	return ret

def process_file(file_path, enforce=0):
	global responses
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
					c.append(tokenize(utt))
					r.append(tokenize(responses[id]))
			# not appending EOS to end of contexts because __eot__ is there at
			# end of each utterance in V2 dataset
	if enforce != 0:
		print("{} Records skipped due to enforce={} is = {}".format(
			fbase, enforce, skipCount
			))
	print("{} -> size = {}".format(fbase, len(y)))
	return y, c, r

if __name__ == "__main__":
	load_vocab(vocab_file)
	build_responses(response_file)

	train_y, train_c, train_r = process_file(train_file)
	valid_y, valid_c, valid_r = process_file(valid_file, 10)
	test_y, test_c, test_r = process_file(test_file, 10)

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
	print("eot:{}\teou:{}".format(eot, eou))
