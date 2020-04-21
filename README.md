# DKN

An implementation of [DKN](https://dl.acm.org/doi/abs/10.1145/3178876.3186175) (*Deep Knowledge-Aware Network for News Recommendation*) in PyTorch.

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

# Copy generated embedding files and datasets to `processed_data` directory
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
Cached value for processed_data\train.txt found in ./cache\85863052e7e0c4dce9e455d97aa59009d091a0dbdb0799c0c066273d81ea00a2, load it directly.
Cached value for processed_data\test.txt found in ./cache\85863052e7e0c4dce9e455d97aa59009aca2bfe2638f71f2720133f987e93cc9, load it directly.
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
Initial result on test dataset, validation loss: 0.684186, validation accuracy: 0.589286
Time 00:00:17, batches 50, current loss 0.676491, average loss: 0.665422
Time 00:00:36, batches 100, current loss 0.636080, average loss: 0.667474
Time 00:00:55, batches 150, current loss 0.667803, average loss: 0.666965
Training data exhausted for 1 times after 163 batches, reuse the dataset.
Time 00:01:26, batches 200, current loss 0.656539, average loss: 0.664880
Time 00:01:44, batches 250, current loss 0.634193, average loss: 0.664927
Time 00:02:04, batches 300, current loss 0.626535, average loss: 0.664399
Training data exhausted for 2 times after 326 batches, reuse the dataset.
Time 00:02:34, batches 350, current loss 0.656598, average loss: 0.663334
Time 00:02:52, batches 400, current loss 0.690482, average loss: 0.662420
Time 00:03:10, batches 450, current loss 0.661049, average loss: 0.662033
Training data exhausted for 3 times after 489 batches, reuse the dataset.
Time 00:03:42, batches 500, current loss 0.648606, average loss: 0.661638
Time 00:03:51, batches 500, validation loss: 0.681755, validation accuracy: 0.584821
Time 00:04:09, batches 550, current loss 0.656039, average loss: 0.660853
Time 00:04:26, batches 600, current loss 0.709801, average loss: 0.660243
Time 00:04:43, batches 650, current loss 0.659273, average loss: 0.660335
Training data exhausted for 4 times after 652 batches, reuse the dataset.
Time 00:05:13, batches 700, current loss 0.645418, average loss: 0.659944
Time 00:05:31, batches 750, current loss 0.643229, average loss: 0.659309
Time 00:05:48, batches 800, current loss 0.669425, average loss: 0.658637
Training data exhausted for 5 times after 815 batches, reuse the dataset.
Time 00:06:19, batches 850, current loss 0.662925, average loss: 0.657923
Time 00:06:37, batches 900, current loss 0.657633, average loss: 0.657215
Time 00:06:55, batches 950, current loss 0.703488, average loss: 0.656871
Training data exhausted for 6 times after 978 batches, reuse the dataset.
Time 00:07:26, batches 1000, current loss 0.563571, average loss: 0.656086
Time 00:07:35, batches 1000, validation loss: 0.674355, validation accuracy: 0.600446
Time 00:07:54, batches 1050, current loss 0.594838, average loss: 0.655463
Time 00:08:12, batches 1100, current loss 0.672383, average loss: 0.655125
Training data exhausted for 7 times after 1141 batches, reuse the dataset.
Time 00:08:42, batches 1150, current loss 0.624101, average loss: 0.654335
Time 00:09:00, batches 1200, current loss 0.619443, average loss: 0.653343
Time 00:09:20, batches 1250, current loss 0.668079, average loss: 0.652750
Time 00:09:39, batches 1300, current loss 0.656603, average loss: 0.652263
Training data exhausted for 8 times after 1304 batches, reuse the dataset.
```

## Credits

- News data, word, entity and context embeddings are from <https://github.com/hwwang55/DKN>.
