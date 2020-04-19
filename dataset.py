from torch.utils.data import Dataset


class DKNDataset(Dataset):
    def __init__(self, config, filepath):
        self.config = config
        self.data = self._read_dataset(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _read_dataset(self, filepath):
        """
        Read and parse the dataset.
        Input example:
        0 CandidateNews:2722,710,981,7251,7000,1263,4013,0,0,0 entity:2735,0,0,0,0,0,0,0,0,0 clickedNews0:7819,1205,2949,5020,417,4087,378,3666,0,0 entity0:0,0,0,0,1316,1316,0,0,0,0 clickedNews1:1218,7526,2209,7643,2860,1757,4487,0,0,0 entity1:0,0,311,0,0,0,0,0,0,0
        Output example:
        [
            {
                "clicked": 0,
                "candidatae_news": {
                    "word": [2722, 710, 981, 7251, 7000, 1263, 4013, 0, 0, 0],
                    "entity":[2735, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                },
                "clicked_news": [
                    {
                        "word": [7819, 1205, 2949, 5020, 417, 4087, 378, 3666, 0, 0],
                        "entity":[0, 0, 0, 0, 1316, 1316, 0, 0, 0, 0]
                    },
                    {
                        "word": [1218, 7526, 2209, 7643, 2860, 1757, 4487, 0, 0, 0],
                        "entity":[0, 0, 311, 0, 0, 0, 0, 0, 0, 0]
                    },
                    padded...
                ]
            }
        ]

        """
        result = []
        f = open(filepath, 'r')

        def parse_helper(text):
            return list(map(int, text.split(':')[-1].split(',')))

        for line in f.readlines():
            if not (len(line.split(' ')) >= 5 and
                    (len(line.split(' ')) - 3) % 2 == 0):
                break
            item_iter = iter(line.split(' '))
            item = {}
            item["clicked"] = int(next(item_iter))
            item["candidatae_news"] = {
                "word": parse_helper(next(item_iter)),
                "entity": parse_helper(next(item_iter))
            }
            item["clicked_news"] = []
            try:
                while True:
                    item["clicked_news"].append({
                        "word":
                        parse_helper(next(item_iter)),
                        "entity":
                        parse_helper(next(item_iter))
                    })
            except StopIteration:
                pass

            # TODO how to pad?
            padding = {
                "word": [0] * self.config.num_words_a_sentence,
                "entity": [0] * self.config.num_words_a_sentence
            }
            repeated_times = self.config.num_clicked_news_a_user - \
                len(item["clicked_news"])
            assert repeated_times >= 0
            item["clicked_news"].extend([padding] * repeated_times)
            result.append(item)

        f.close()

        return result
