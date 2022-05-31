# R-Transformer


## Quick start

Prepare virtual environment:

```console
$ python3 -m venv venv
$ source ./venv/bin/activate
```

Install required packages:

```console
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
```

## Usage

```console
(venv) $ python main.py
```

```
usage: main.py [-h] [--model MODEL] [--em EM] [--hu HU] [--layers LAYERS] [--lr LR] [--clip CLIP] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--seqlen SEQLEN] [--dropout DROPOUT] [--seed SEED]
               [--log_interval LOG_INTERVAL]

Neural Language Model

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of network (GRU/LSTM/R-Transformer)
  --em EM               size of word embeddings
  --hu HU               number of hidden units per layer
  --layers LAYERS       number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       number of training epochs
  --batch_size BATCH_SIZE
                        batch size
  --seqlen SEQLEN       sequence length
  --dropout DROPOUT     dropout
  --seed SEED           random seed
  --log_interval LOG_INTERVAL
                        report interval (in batches)
```