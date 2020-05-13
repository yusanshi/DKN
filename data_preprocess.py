from config import Config
import pandas as pd
from os import path


def clean(dir_path):
    print(f"Clean up {path.join(dir_path, 'behaviors.tsv')}")
    behaviors = pd.read_table(path.join(dir_path, 'behaviors.tsv'), header=None, usecols=[
        2, 3], names=['clicked_news', 'impressions'])
    behaviors.impressions = behaviors.impressions.str.split()
    behaviors = behaviors.explode('impressions').reset_index(drop=True)
    behaviors['candidate_news'], behaviors['clicked'] = behaviors.impressions.str.split(
        '-').str
    behaviors.dropna(inplace=True)
    behaviors.to_csv(
        path.join(dir_path, 'behaviors_cleaned.tsv'), sep='\t', index=False, columns=['clicked_news', 'candidate_news', 'clicked'])

    print(f"Clean up {path.join(dir_path, 'news.tsv')}")
    news = pd.read_table(path.join(dir_path, 'news.tsv'), header=None, usecols=[
                         0, 3, 4, 6], names=['id', 'title', 'abstract', 'entities'])
    for row in news.itertuples():
        if pd.isnull(row.abstract):
            news.at[row.Index, 'abstract'] = ''
    news['text'] = news['title'] + '. ' + news['abstract']
    news.to_csv(
        path.join(dir_path, 'news_cleaned.tsv'), sep='\t', index=False, columns=['id', 'text', 'entities'])


if __name__ == '__main__':
    print('Clean up data in `merged`')
    clean('./data/merged')
