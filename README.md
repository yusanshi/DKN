# DKN

An implementation of [DKN](https://dl.acm.org/doi/abs/10.1145/3178876.3186175) (_Deep knowledge-aware network for news recommendation_) in PyTorch.

**[WIP] I'm still debugging it.**

## Get started

Basic setup.

```bash
git clone https://github.com/yusanshi/DKN
cd DKN
pip3 install -r requirements.txt
```

Prepare word, entity and context embedding. Taken from <https://github.com/hwwang55/DKN>.

```bash
# Credits: this dkn_data.tgz file is from
# https://github.com/hwwang55/DKN/tree/90a188021a82ddaadffc44f6d87e1e72b1c3db9a/data
wget -O - https://yun.yusanshi.com/dkn_data.tgz | tar -xjvf -
cd data/news
# Before running news_preprocess.py, you may want to edit
# MAX_TITLE_LENGTH and WORD_EMBEDDING_DIM in it
python3 news_preprocess.py
cd ../kg
python3 prepare_data_for_transx.py
# Note: you can also choose other KGE methods
cd Fast-TransX/transE/
g++ transE.cpp -o transE -pthread -O3 -march=native
./transE
cd ../..
# Before running kg_preprocess.py, you may want to edit
# ENTITY_EMBEDDING_DIM in it
python3 kg_preprocess.py

# Copy generated embeddings file and datasets to `embeddings` directory
cd ../..
mkdir processed_data
# If you have chosen TransE and 50 for word and entity embedding, for example
cp data/news/word_embeddings_50.npy processed_data/word.npy
cp data/kg/entity_embeddings_TransE_50.npy processed_data/entity.npy
cp data/kg/context_embeddings_TransE_50.npy processed_data/context.npy
cp data/news/train.txt processed_data/
cp data/news/test.txt processed_data/
```

Run.

```bash
# Before training, you may want to edit config.py file.
# If you have chosen different values for MAX_TITLE_LENGTH,
# WORD_EMBEDDING_DIM and ENTITY_EMBEDDING_DIM
# other than default value 10, 50, 50,
# you should also modify them in config.py accordingly
python3 main.py
```

Example output.

```
Using device: cuda:0
Cached value for processed_data\train.txt not found, wait patiently for it to be parsed.
Parsed data for processed_data\train.txt was cached in ./cache\03a3226909139e079d11d36f7c43356ed091a0dbdb0799c0c066273d81ea00a2 for future use.
Cached value for processed_data\test.txt not found, wait patiently for it to be parsed.
Parsed data for processed_data\test.txt was cached in ./cache\03a3226909139e079d11d36f7c43356eaca2bfe2638f71f2720133f987e93cc9 for future use.
Load dataset with train size 10401 and test size 462.
DKN(
  (kcnn): KCNN(
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
Time 00:00:17, batches 40, current loss 0.717675507068634, average loss: 0.6699299409985542
Time 00:00:34, batches 80, current loss 0.6553049087524414, average loss: 0.668458455055952
Time 00:00:52, batches 120, current loss 0.612324595451355, average loss: 0.6689336990316709
Time 00:01:10, batches 160, current loss 0.661110520362854, average loss: 0.6704976052045822
Training data exhausted for 1 times after 163 batches. Reuse the dataset.
Time 00:01:39, batches 200, current loss 0.6319454908370972, average loss: 0.670027393492023
Time 00:01:55, batches 240, current loss 0.658223569393158, average loss: 0.6681504124876845
Time 00:02:12, batches 280, current loss 0.6902536153793335, average loss: 0.6688428078928301
Time 00:02:28, batches 320, current loss 0.6598144769668579, average loss: 0.6680968566374346
Training data exhausted for 2 times after 326 batches. Reuse the dataset.
Time 00:02:57, batches 360, current loss 0.7246805429458618, average loss: 0.6676784236004899
Time 00:03:15, batches 400, current loss 0.7002042531967163, average loss: 0.6675169325653633
Time 00:03:32, batches 440, current loss 0.6508539915084839, average loss: 0.6669909394767186
Time 00:03:50, batches 480, current loss 0.6525683403015137, average loss: 0.6669516747965474
Training data exhausted for 3 times after 489 batches. Reuse the dataset.
Time 00:04:12, batches 500, validation loss: 0.6744713698114667
Time 00:04:30, batches 520, current loss 0.6331638693809509, average loss: 0.6669475692153207
Time 00:04:48, batches 560, current loss 0.6538596153259277, average loss: 0.6668522126798784
Time 00:05:06, batches 600, current loss 0.6618661284446716, average loss: 0.666427685188089
Time 00:05:26, batches 640, current loss 0.6587792038917542, average loss: 0.6659404841285298
Training data exhausted for 4 times after 652 batches. Reuse the dataset.
Time 00:05:56, batches 680, current loss 0.6696106195449829, average loss: 0.6654807592813785
Time 00:06:14, batches 720, current loss 0.6815027594566345, average loss: 0.6651551151575323
Time 00:06:31, batches 760, current loss 0.6893236637115479, average loss: 0.6650966211916909
Time 00:06:48, batches 800, current loss 0.6396827101707458, average loss: 0.6646398055793052
Training data exhausted for 5 times after 815 batches. Reuse the dataset.
Time 00:07:18, batches 840, current loss 0.6417379379272461, average loss: 0.6645475499644251
Time 00:07:35, batches 880, current loss 0.6852426528930664, average loss: 0.6640259487969534
Time 00:07:52, batches 920, current loss 0.6498686671257019, average loss: 0.6642090747916634
```

## Credits

- News data, word, entity and context embeddings are from <https://github.com/hwwang55/DKN>.

