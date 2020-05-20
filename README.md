***Integrated into <https://github.com/yusanshi/NewsRecommendation>. This repository is currently read-only.***

---

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
unzip MINDsmall_dev.zip -d test
rm MINDsmall_*.zip

# Preprocess data into appropriate format
cd ..
python3 src/data_preprocess.py
# Remember you shoud modify `num_word_tokens` in `src/config.py` by the output of `src/data_preprocess.py`
```

Run.

```bash
python3 src/train.py
python3 src/inference.py
python3 src/evaluate.py

# or use `run.sh` to compare the result with or without context embedding, attention mechanism.

chmod +x run.sh
./run.sh
```

You can visualize the training loss and accuracy with TensorBoard.

```bash
tensorboard --logdir=runs
```
Note the metrics in validation will differ greatly with final result on evaluation set. You should use it for reference only.

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
