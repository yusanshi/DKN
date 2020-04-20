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

## Credits

- News data, word, entity and context embeddings are from <https://github.com/hwwang55/DKN>.

