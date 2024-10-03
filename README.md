# DAM-Updated Implementation on Ubuntu Corpus

This repository contains the updated implementation of Multi-Turn Response
Selection for Chatbots with Deep Attention Matching Network (DAM)

All the python files in this repository must be run with python 2.7, and 
tensorflow 1.10 or a similar version.

## Preprocessing

1. place the dataset in `data/ubuntu/`. For example, train.txt will go to:
	`data/ubuntu/train.txt`.
2. generate word embeddings by running `python word_embedding.py`
3. generate data.pkl file by running `python data.py` 
4. note the value of `eos` printed by `data.py`. This value will be different
	for each dataset (for each `vocab.txt` to be precise)

> [!NOTE]
> The evaluation function is hardcoded to expect 1 valid response, and 9
> invalid responses, totalling 10 responses. However there are 118-117 such
> lines in UDC V2 `test.txt` and `valid.txt` where there are 8 or 9 responses
> total. We have made it so the `data.py` script will ignore such lines.

## Training

First enter the noted value of `eou` from `data.py` into `main.py`:

```python
conf = {
	...
	"_EOS_": valueHere
	...
}
```

This value of `_EOS_` should match that of `__eou__` in `vocab.txt`.

Make sure the `main.py` file has the following lines at end not commented:

```python
model = net.Net(conf)
train.train(conf, model)
```

Run the `main.py` file as: `python main.py`

Or if you would like to save the training log:

```bash
python -u main.py 2>&1 | tee trainlog.txt
```

## Evaluation

Set the `init_model` conf paramter in `main.py` to the path of saved checkpoint.

For example:

```python
conf = {
	"data_path": "./data/ubuntu/data.pkl",
	"save_path": "./output/ubuntu/temp/",
	"word_emb_init": "./data/word_embedding.pkl",
	"init_model": "./output/ubuntu/temp/model.ckpt.1"
	...
}
```

Then comment the following line in `main.py`:

```python
train.train(conf, model)
```

And uncomment the following line:

```python
test.test(conf, model)
```

Run the evaluation as `python main.py`

Or if you would like to save the testing log:

```bash
python -u main.py 2>&1 | tee testlog.txt
```

Run the following to calculate Top-1 Precision:

```bash
python precision.py
```

## Acknowledgments

[DAM GitHub Repository](https://github.com/baidu/Dialogue/tree/master/DAM) \
[Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://aclanthology.org/P18-1103.pdf)
