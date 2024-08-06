import pickle
import sys

if len(sys.argv) < 2:
	print("data pickle file path missing")
	exit(1)
datapath = sys.argv[1]
prefix = "./"
if len(sys.argv) > 2:
	prefix = sys.argv[2]

f = open(datapath, 'rb')
train_data, val_data, test_data = pickle.load(f)
f.close()
with open(prefix + "dumptrain.txt", "w") as f:
	f.write(str(train_data))
with open(prefix + "dumpvalid.txt", "w") as f:
	f.write(str(val_data))
with open(prefix + "dumptest.txt", "w") as f:
	f.write(str(test_data))
print('finish loading data')
