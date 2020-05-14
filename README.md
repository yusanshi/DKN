# DKN

An implementation of [DKN](https://dl.acm.org/doi/abs/10.1145/3178876.3186175) (_Deep Knowledge-Aware Network for News Recommendation_) in PyTorch.

## Get started

Basic setup.

```bash
git clone https://github.com/yusanshi/DKN
cd DKN
pip3 install -r requirements.txt
```

Download the dataset.

```bash
mkdir data && cd data

# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
sudo apt install unzip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d validate
rm MINDsmall_*.zip

# Merge train and validate dataset (currently the dataset is small so we do so to enlarge it)
mkdir merged
sort -u train/behaviors.tsv validate/behaviors.tsv > merged/behaviors.tsv
sort -u train/news.tsv validate/news.tsv > merged/news.tsv
sort -u train/entity_embedding.vec validate/entity_embedding.vec > merged/entity_embedding.vec

# Preprocess data in `merged` into appropriate format
cd ..
python3 src/data_preprocess.py
```

Run.

```bash
python3 src/main.py

# or use `run.sh` to compare the result with or without context embedding, attention mechanism.

chmod +x run.sh
./run.sh
```

You can visualize the result with TensorBoard.

```bash
tensorboard --logdir=runs
```

Example output.

```
Using device: cuda:0
Context: False, Attention: True
2020-05-14 20:22:43.266593: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
2020-05-14 20:22:43.268043: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
Load dataset with train size 814795 and test size 203699.
DKN(
  (kcnn): KCNN(
    (word_embedding): Embedding(20674, 100)
    (conv_filters): ModuleDict(
      (2): Conv2d(2, 50, kernel_size=(2, 100), stride=(1, 1))
      (3): Conv2d(2, 50, kernel_size=(3, 100), stride=(1, 1))
      (4): Conv2d(2, 50, kernel_size=(4, 100), stride=(1, 1))
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
Checking loss and accuracy: 100%|█████████████████████████████████████████████████| 100/100 [04:00<00:00,  2.41s/it]
Initial result on test dataset, validation loss: 0.706865, validation accuracy: 0.358867
Time 00:02:05, batches 50, current loss 0.592621, average loss: 0.624477
Time 00:04:11, batches 100, current loss 0.581295, average loss: 0.612426
Time 00:06:20, batches 150, current loss 0.605994, average loss: 0.605928
Time 00:08:25, batches 200, current loss 0.610041, average loss: 0.600768
Time 00:10:30, batches 250, current loss 0.621125, average loss: 0.597274
Time 00:12:29, batches 300, current loss 0.562352, average loss: 0.593654
Checking loss and accuracy: 100%|█████████████████████████████████████████████████| 100/100 [03:41<00:00,  2.21s/it]
Time 00:16:10, batches 300, validation loss: 0.583402, validation accuracy: 0.703867

......
```

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
