from config import Config
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from os import path
import re


def clean_dataset(behaviors_source, behaviors_target, news_source,
                  news_target):
    """
    Remove unnecessary information in MIND dataset
    Args:
        behaviors_source: behaviors.tsv file in MIND dataset
            example:
                U1	11/15/2019 7:28:25 AM	N8240 N61893 N17763 N15442 N45269 N27247 N2807 N64305 N58244 N62099 N1211 N7544	N18774-1 N55354-0
        behaviors_target:
            example:
                clicked_news	candidate_news	clicked
                N12142 N55361 N42151 N5313 N38326 N60863 N32104 N36290 N65 N43756 N1686 N54143 N64745 N54637 N56978 N26686 N31733 N31851 N32288 N57578 N39175 N22904 N9874 N7544 N7228 N61247 N39144 N28742 N10369 N12912 N29465 N38587 N49827 N35943	N11611	0

        news_source: news.tsv file in MIND dataset
            example:
                N1	lifestyle	lifestylefamily	College gymnast dies following practice accident in Connecticut		https://www.msn.com/en-us/lifestyle/lifestylefamily/college-gymnast-dies-following-practice-accident-in-connecticut/ar-BBWBre3?ocid=chopendata	[{"Label": "Connecticut", "Type": "G", "WikidataId": "Q779", "Confidence": 0.999, "OccurrenceOffsets": [54], "SurfaceForms": ["Connecticut"]}]
        news_target:
            example:
                id	title	entities
                N1	College gymnast dies following practice accident in Connecticut	"[{""Label"": ""Connecticut"", ""Type"": ""G"", ""WikidataId"": ""Q779"", ""Confidence"": 0.999, ""OccurrenceOffsets"": [54], ""SurfaceForms"": [""Connecticut""]}]"
    """
    print(f"Clean up {behaviors_source}")
    behaviors = pd.read_table(behaviors_source,
                              header=None,
                              usecols=[2, 3],
                              names=['clicked_news', 'impressions'])
    behaviors.impressions = behaviors.impressions.str.split()
    behaviors = behaviors.explode('impressions').reset_index(drop=True)
    behaviors['candidate_news'], behaviors[
        'clicked'] = behaviors.impressions.str.split('-').str
    behaviors.clicked_news.fillna('', inplace=True)
    behaviors.to_csv(behaviors_target,
                     sep='\t',
                     index=False,
                     columns=['clicked_news', 'candidate_news', 'clicked'])

    print(f"Clean up {news_source}")
    news = pd.read_table(news_source,
                         header=None,
                         usecols=[0, 3, 6],
                         names=['id', 'title', 'entities'])
    news.to_csv(news_target, sep='\t', index=False)


def balance(source, target, true_false_division_range):
    """
    Args:
        source: file path of original behaviors tsv file
        target: file path of balanced behaviors tsv file
        true_false_division_range: (low, high), len(true_part) / len(false_part) will be within the range
    """
    low = true_false_division_range[0]
    high = true_false_division_range[1]
    assert low <= high
    original = pd.read_table(source)
    true_part = original[original['clicked'] == 1]
    false_part = original[original['clicked'] == 0]
    if len(true_part) / len(false_part) < low:
        print(
            f'Drop {len(false_part) - int(len(true_part) / low)} from false part'
        )
        false_part = false_part.sample(n=int(len(true_part) / low))
    elif len(true_part) / len(false_part) > high:
        print(
            f'Drop {len(true_part) - int(len(false_part) * high)} from true part'
        )
        true_part = true_part.sample(n=int(len(false_part) * high))

    balanced = pd.concat([true_part,
                          false_part]).sample(frac=1).reset_index(drop=True)
    balanced.to_csv(target, sep='\t', index=False)


def parse_news(source, target, word2int_path, entity2int_path, mode):
    """
    Args:
        source: path of tsv file as input
            example:
            id	title	entities
            N1	College gymnast dies following practice accident in Connecticut	"[{""Label"": ""Connecticut"", ""Type"": ""G"", ""WikidataId"": ""Q779"", ""Confidence"": 0.999, ""OccurrenceOffsets"": [54], ""SurfaceForms"": [""Connecticut""]}]"

        target: path of tsv file
            example:
                id	title	entities
                N1	[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if mode == 'train':
            word2int_path: where to load
            entity2int_path: where to load
        elif mode == 'test':
            word2int_path: where to load
            entity2int_path: where to load
    """
    def clean_text(text):
        return re.sub(r'[^a-zA-Z ]', '', text).lower().strip()

    if mode == 'train':
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        news = pd.read_table(source)
        news.entities.fillna('[]', inplace=True)
        parsed_news = pd.DataFrame(columns=['id', 'title', 'entities'])

        with tqdm(total=len(news), desc="Counting words and entities") as pbar:
            for row in news.itertuples(index=False):
                for w in clean_text(row.title).split():
                    if w not in word2freq:
                        word2freq[w] = 1
                    else:
                        word2freq[w] += 1
                for e in json.loads(row.entities):
                    # Count occurrence time within title
                    times = len(
                        list(
                            filter(lambda x: x < len(row.title),
                                   e['OccurrenceOffsets']))) * e['Confidence']
                    if times > 0:
                        if e['WikidataId'] not in entity2freq:
                            entity2freq[e['WikidataId']] = times
                        else:
                            entity2freq[e['WikidataId']] += times
                pbar.update(1)

        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= Config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        with tqdm(total=len(news), desc="Parsing words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id, [0] * Config.num_words_a_news,
                    [0] * Config.num_words_a_news
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]
                try:
                    for i, w in enumerate(clean_text(row.title).split()):
                        if w in word2int:
                            new_row[1][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)
        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_word_tokens` in `src/config.py` into 1 + {len(word2int)}'
        )
        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
    elif mode == 'test':
        news = pd.read_table(source)
        news.entities.fillna('[]', inplace=True)
        parsed_news = pd.DataFrame(columns=['id', 'title', 'entities'])

        word2int = dict(pd.read_table(word2int_path).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        word_total = 0
        word_missed = 0

        with tqdm(total=len(news), desc="Parsing words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id, [0] * Config.num_words_a_news,
                    [0] * Config.num_words_a_news
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]
                try:
                    for i, w in enumerate(clean_text(row.title).split()):
                        word_total += 1
                        if w in word2int:
                            new_row[1][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)
        print(f'Out-of-Vocabulary rate: {word_missed/word_total:.4f}')
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


def transform_entity_embedding(source, target, entity2int_path):
    """
    Args:
        source: path of embedding file
            example:
                Q100	-0.075855	-0.164252	0.128812	-0.022738	-0.127613	-0.160166	0.138481	-0.135568	0.117921	-0.003037	0.127557	0.142511	0.084117	-0.004320	-0.090240	0.009786	0.013588	0.003356	-0.066014	-0.098590	-0.088168	0.055409	-0.004417	0.118718	-0.035986	-0.010574	0.060249	0.064847	0.106534	0.015566	-0.077538	0.027226	0.040080	-0.132547	-0.015346	0.048049	-0.139377	-0.152344	-0.050292	0.022452	-0.122296	-0.026120	0.008042	-0.059975	-0.132461	-0.102174	-0.122510	0.008978	-0.011055	0.114250	-0.109533	0.012790	0.120282	0.031591	0.043915	-0.014192	-0.000558	-0.009249	-0.023576	-0.054018	-0.143273	0.131889	0.090060	0.056647	0.062646	-0.198711	-0.162954	-0.160493	-0.042409	-0.043214	-0.117995	-0.160036	0.090786	0.129228	-0.118732	-0.022712	-0.001741	0.156582	0.011148	0.027286	0.047676	0.002435	0.019395	0.140718	0.139035	-0.081709	0.034342	0.059993	-0.141031	-0.072964	-0.104429	0.084221	0.036348	-0.128924	-0.228023	-0.180280	-0.025696	-0.141512	0.037383	0.085674
        target: path of transformed embedding file in numpy format
        entity2int_path
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:,
                                                       1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector'
                                         ]].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(entity_embedding, entity2int,
                         on='entity').sort_values('int')
    # TODO in fact, some entity in entity2int cannot be found in entity_embedding
    # see https://github.com/msnews/MIND/issues/2
    entity_embedding_transformed = np.zeros(
        (len(entity2int) + 1, Config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)


def transform2json(source, target):
    """
    Transform bahaviors file in tsv to json for later evaluation
    Args:
        TODO
        source:
        target:
    """
    behaviors = pd.read_table(
        source,
        header=None,
        names=['uid', 'time', 'clicked_news', 'impression'])
    f = open(target, "w")
    with tqdm(total=len(behaviors), desc="Transforming tsv to json") as pbar:
        for row in behaviors.itertuples(index=False):
            item = {}
            item['uid'] = row.uid[1:]
            item['time'] = row.time
            item['impression'] = {
                x.split('-')[0][1:]: int(x.split('-')[1])
                for x in row.impression.split()
            }
            f.write(json.dumps(item) + '\n')

            pbar.update(1)

    f.close()


if __name__ == '__main__':
    train_dir = './data/train'
    test_dir = './data/test'

    print('Process data for training')

    print('Clean up data')
    clean_dataset(path.join(train_dir, 'behaviors.tsv'),
                  path.join(train_dir, 'behaviors_cleaned.tsv'),
                  path.join(train_dir, 'news.tsv'),
                  path.join(train_dir, 'news_cleaned.tsv'))

    print('Balance data')
    balance(path.join(train_dir, 'behaviors_cleaned.tsv'),
            path.join(train_dir, 'behaviors_cleaned_balanced.tsv'), (1 / 2, 2))

    print('Parse news')
    parse_news(path.join(train_dir, 'news_cleaned.tsv'),
               path.join(train_dir, 'news_with_entity.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='train')

    print('Transform entity embeddings')
    transform_entity_embedding(path.join(train_dir, 'entity_embedding.vec'),
                               path.join(train_dir, 'entity_embedding.npy'),
                               path.join(train_dir, 'entity2int.tsv'))

    print('\nProcess data for evaluation')

    print('Transform test data')
    transform2json(path.join(test_dir, 'behaviors.tsv'),
                   path.join(test_dir, 'truth.json'))

    print('Clean up data')
    clean_dataset(path.join(test_dir, 'behaviors.tsv'),
                  path.join(test_dir, 'behaviors_cleaned.tsv'),
                  path.join(test_dir, 'news.tsv'),
                  path.join(test_dir, 'news_cleaned.tsv'))

    print('Parse news')
    parse_news(path.join(test_dir, 'news_cleaned.tsv'),
               path.join(test_dir, 'news_with_entity.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='test')
