# DKN

An implementation of [DKN](https://dl.acm.org/doi/abs/10.1145/3178876.3186175) (*Deep Knowledge-Aware Network for News Recommendation*) in PyTorch.

## Get started

Basic setup.

```bash
git clone --recursive https://github.com/yusanshi/DKN
cd DKN
pip3 install -r requirements.txt
```


Download the dataset and train knowledge embedding yourself.

```bash
wget -O - https://yusanshi.com/dkn_data.tgz | tar -xjvf -
# Preprocess data
python3 data_preprocess.py
# Train the knowledge graph embedding
python3 knowledge_graph.py train
# Generate embedding files from trained model (transe.ckpt)
python3 knowledge_graph.py generate
```

You can also simply use my trained version.

```bash
wget -O - https://yusanshi.com/dkn_data_trained.tgz | tar -xjvf -
```

Run.

```bash
python3 main.py

# or use `run.sh` to compare the result with or without context embedding, attention mechanism.

chmod +x run.sh
./run.sh
```

You can visualize the result with TensorBoard.
```bash
tensorboard --logdir=runs
```

Train loss comparasion.

**TODO**

Example output.

```
Using device: cuda:0
Context: True, Attention: True
Cached value for data/news/news.txt found in ./cache/6eaf15e083812ce71fdabccf7e1ea7e9b6d85a0a90c7e464e5528fd3d41b44fe, load it directly.
Load dataset with train size 7494 and test size 3212.
DKN(
  (kcnn): KCNN(
    (word_embedding): Embedding(7948, 50)
    (conv_filters): ModuleDict(
      (2): Conv2d(3, 50, kernel_size=(2, 50), stride=(1, 1))
      (3): Conv2d(3, 50, kernel_size=(3, 50), stride=(1, 1))
      (4): Conv2d(3, 50, kernel_size=(4, 50), stride=(1, 1))
    )
  )
  (attention): Attention(
    (dnn): Sequential(
      (0): Linear(in_features=300, out_features=16, bias=True)
      (1): Linear(in_features=16, out_features=1, bias=True)
    )
  )
  (dnn): Sequential(
    (0): Linear(in_features=300, out_features=16, bias=True)
    (1): Linear(in_features=16, out_features=1, bias=True)
  )
)
Initial result on test dataset, validation loss: 0.695761, validation accuracy: 0.472187
Time 00:00:19, batches 50, current loss 0.663087, average loss: 0.669179
Time 00:00:37, batches 100, current loss 0.623621, average loss: 0.668357
Training data exhausted for 1 times after 118 batches, reuse the dataset.
Time 00:00:57, batches 150, current loss 0.629727, average loss: 0.663313
Time 00:01:16, batches 200, current loss 0.707039, average loss: 0.657151
Training data exhausted for 2 times after 236 batches, reuse the dataset.
Time 00:01:37, batches 250, current loss 0.475895, average loss: 0.650020
Time 00:01:55, batches 300, current loss 0.566996, average loss: 0.634187
Time 00:02:15, batches 350, current loss 0.634617, average loss: 0.623723
Training data exhausted for 3 times after 354 batches, reuse the dataset.
Time 00:02:35, batches 400, current loss 0.359774, average loss: 0.600908
Time 00:02:54, batches 450, current loss 0.501074, average loss: 0.581651
Training data exhausted for 4 times after 472 batches, reuse the dataset.
Time 00:03:13, batches 500, current loss 0.292432, average loss: 0.562351
Time 00:03:30, batches 500, validation loss: 0.870142, validation accuracy: 0.582500
Time 00:03:50, batches 550, current loss 0.426439, average loss: 0.541557
Training data exhausted for 5 times after 590 batches, reuse the dataset.
Time 00:04:10, batches 600, current loss 0.185671, average loss: 0.524622
Time 00:04:30, batches 650, current loss 0.261221, average loss: 0.501985
Time 00:04:49, batches 700, current loss 0.355073, average loss: 0.487202
Training data exhausted for 6 times after 708 batches, reuse the dataset.
```

## Credits
- Knowledge embedding training by [OpenKE](https://github.com/thunlp/OpenKE).
- Dataset based on <https://github.com/hwwang55/DKN/tree/0a84abba033cd1a873daaa90c8fd6878e5875e64/data>.
