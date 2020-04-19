# DKN

An implementation of [DKN](https://dl.acm.org/doi/abs/10.1145/3178876.3186175) (_Deep knowledge-aware network for news recommendation_) in PyTorch.

## Get started

Basic setup.

```bash
git clone https://github.com/yusanshi/DKN
cd DKN
pip3 install -r requirements.txt
```

Prepare word, entity and context embedding. Taken from <https://github.com/hwwang55/DKN>.

```bash
wget -O - https://yun.yusanshi.com/dkn_data.tgz | tar -xjvf -
cd data/news
python3 news_preprocess.py
cd ../kg
python3 prepare_data_for_transx.py
# Note: you can also choose other KGE methods
cd Fast-TransX/transE/
g++ transE.cpp -o transE -pthread -O3 -march=native
./transE
cd ../..
python3 kg_preprocess.py
```

Run.

```
cd ../..
python3 main.py
```

## Credits

- News data, entity embedding and context embedding are from <https://github.com/hwwang55/DKN> (See `./thirdparty/hwwang55`).
