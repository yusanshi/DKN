from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import Config


class DKNDataset(Dataset):
    def __init__(self, behaviors_path, news_with_entity_path):
        """
        Args:
            behaviors_path: path of behaviors tsv file
                example:
                    clicked_news	candidate_news	clicked
                    N12142 N55361 N42151 N5313 N38326 N60863 N32104 N36290 N65 N43756 N1686 N54143 N64745 N54637 N56978 N26686 N31733 N31851 N32288 N57578 N39175 N22904 N9874 N7544 N7228 N61247 N39144 N28742 N10369 N12912 N29465 N38587 N49827 N35943	N11611	0

            news_with_entity_path: path of news_with_entity, map news id to title and entities
                example:
                    id	title	entities
                    N1	[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        super(Dataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path)
        self.news_with_entity = pd.read_table(news_with_entity_path,
                                              index_col='id',
                                              converters={
                                                  'title': literal_eval,
                                                  'entities': literal_eval
                                              })

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        """
        example:
            {
                clicked: 0
                candidate_news:
                    {
                        "word": [0] * num_words_a_news,
                        "entity": [0] * num_words_a_news
                    }
                clicked_news:
                    [
                        {
                            "word": [0] * num_words_a_news,
                            "entity": [0] * num_words_a_news
                        } * num_clicked_news_a_user
                    ]
            }
        """
        def news2dict(news, df):
            return {
                "word": df.loc[news].title,
                "entity": df.loc[news].entities
            } if news in df.index else {
                "word": [0] * Config.num_words_a_news,
                "entity": [0] * Config.num_words_a_news
            }
            # TODO the else part is unexpected and is due to error in dataset format

        item = {}
        row = self.behaviors.iloc[idx]
        item["clicked"] = row.clicked
        item["candidate_news"] = news2dict(row.candidate_news,
                                           self.news_with_entity)
        item["clicked_news"] = [
            news2dict(x, self.news_with_entity)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]
        padding = {
            "word": [0] * Config.num_words_a_news,
            "entity": [0] * Config.num_words_a_news
        }
        repeated_times = Config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)

        return item
