from torch.utils.data import Dataset
import pandas as pd
import os
import hashlib
import pickle


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DKNDataset(Dataset):
    def __init__(self, config, filepath):
        self.config = config

        # Use md5 as the identifier, cache the result of _read_dataset
        # Load the result directly if exists
        hash = hashlib.md5(str(vars(
            self.config)).encode('utf-8')).hexdigest() + md5(filepath)
        cache_dir = './cache'
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        pickle_filepath = os.path.join(cache_dir, hash)
        if os.path.isfile(pickle_filepath):
            print(
                f"Cached value for {filepath} found in {pickle_filepath}, load it directly."
            )
            with open(pickle_filepath, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print(
                f"Cached value for {filepath} not found, wait patiently for it to be parsed."
            )
            self.data = self._read_dataset(filepath)
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(self.data, f)
            print(
                f"Parsed data for {filepath} was cached in {pickle_filepath} for future use."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _read_dataset(self, filepath):
        """
        Read and parse the dataset.

        Input example:
        <user_id> <news_word> <news_entity> <clicked>
        19800	2910,6310,321,2432,7972,1377,2115,0,0,0	0,0,0,0,3533,3533,3533,0,0,0	0
        112800	7050,81,7973,48,334,244,1377,4923,0,0	3534,3534,3534,0,0,0,0,0,0,0	1
        112800	7753,7050,81,7973,2078,1460,244,1377,4923,0	0,3534,3534,3534,0,0,0,0,0,0	1
        112800	4142,3143,92,2690,6810,134,5803,1538,7974,7975	811,0,0,0,0,0,0,0,0,0	1

        Output format:
        [
            {
                "clicked": 0,
                "candidatae_news": {
                    "word": [0] * num_words_a_sentence,
                    "entity": [0] * num_words_a_sentence
                },
                "clicked_news": [
                    {
                        "word": [0] * num_words_a_sentence,
                        "entity": [0] * num_words_a_sentence
                    },
                    padded to <num_clicked_news_a_user>
                ]
            },
            ...
        ]

        """

        df = pd.read_table(
            filepath,
            header=None,
            names=['user_id', 'news_word', 'news_entity', 'clicked'])

        result = []
        for user_id in set(df['user_id']):
            for x in df[df["user_id"] == user_id].itertuples(index=False):
                item = {}
                item["clicked"] = x.clicked
                item["candidatae_news"] = {
                    "word": list(map(int, x.news_word.split(','))),
                    "entity": list(map(int, x.news_entity.split(',')))
                }
                available_df = df[(df["user_id"] == user_id)
                                  & (df["clicked"] == 1) &
                                  (df["news_word"] != x.news_word)]
                item["clicked_news"] = [{
                    "word":
                    list(map(int, y.news_word.split(','))),
                    "entity":
                    list(map(int, y.news_entity.split(',')))
                } for y in (
                    available_df.sample(
                        n=self.config.num_clicked_news_a_user).itertuples(
                            index=False) if len(available_df) >= self.config.
                    num_clicked_news_a_user else available_df.itertuples(
                        index=False))]
                padding = {
                    "word": [0] * self.config.num_words_a_sentence,
                    "entity": [0] * self.config.num_words_a_sentence
                }
                repeated_times = self.config.num_clicked_news_a_user - \
                    len(item["clicked_news"])
                assert repeated_times >= 0
                item["clicked_news"].extend([padding] * repeated_times)
                result.append(item)

        return result
