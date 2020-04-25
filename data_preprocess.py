# Credit
# Based on hwwang55@GitHub's work
# https://github.com/hwwang55/DKN/blob/6fa6098613fec7a16a79ce9fd87126521a7a9de1/data/news/news_preprocess.py

from config import Config
import re
import pandas as pd

PATTERN1 = re.compile('[^A-Za-z]')
PATTERN2 = re.compile('[ ]{2,}')
word2freq = {}
word2index = {}
entity2freq = {}
entity2name = {}


def count_word_and_entity_freq(file):
    """
    Count the frequency of words and entities in news titles in the file
    :param files: filename
    :return: None
    """
    reader = open(file, encoding='utf-8')
    for line in reader:
        array = line.strip().split('\t')
        news_title = array[1]
        entities = array[3]

        # count word frequency
        for s in news_title.split(' '):
            if s not in word2freq:
                word2freq[s] = 1
            else:
                word2freq[s] += 1

        # count entity frequency
        for s in entities.split(';'):
            entity_id = s[:s.index(':')]
            entity_name = s[s.index(':') + 1:]
            entity2name[int(entity_id)] = entity_name
            if entity_id not in entity2freq:
                entity2freq[entity_id] = 1
            else:
                entity2freq[entity_id] += 1

    reader.close()


def construct_word2id():
    """
    Allocate each valid word a unique index (start from 1)
    :return: None
    """
    cnt = 1  # 0 is for dummy word
    for w, freq in word2freq.items():
        if freq >= Config.word_freq_threshold:
            word2index[w] = cnt
            cnt += 1
    print('word size: %d' % len(word2index))


def get_local_word2entity(entities):
    """
    Given the entities information in one line of the dataset, construct a map from word to entity
    E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry': id_1,
    'potter': id_1, 'england': id_2}
    :param entities: entities information in one line of the dataset
    :return: a local map from word to entity index
    """
    local_map = {}

    for entity_pair in entities.split(';'):
        entity_id = entity_pair[:entity_pair.index(':')]
        if entity2freq[entity_id] >= Config.entity_freq_threshold:
            entity_name = entity_pair[entity_pair.index(':') + 1:]

            # remove non-character word and transform words to lower case
            entity_name = PATTERN1.sub(' ', entity_name)
            entity_name = PATTERN2.sub(' ', entity_name).lower()

            # constructing map: word -> entity_index
            for w in entity_name.split(' '):
                local_map[w] = entity_id

    return local_map


def encoding_title(title, entities):
    """
    Encoding a title according to word2index map
    :param title: a piece of news title
    :param entities: entities contained in the news title
    :return: encodings of the title with respect to word and entity, respectively
    """
    local_map = get_local_word2entity(entities)

    array = title.split(' ')
    word_encoding = ['0'] * Config.num_words_a_sentence
    entity_encoding = ['0'] * Config.num_words_a_sentence

    point = 0
    for s in array:
        if s in word2index:
            word_encoding[point] = str(word2index[s])
            if s in local_map:
                entity_encoding[point] = str(local_map[s])
            point += 1
        if point == Config.num_words_a_sentence:
            break
    word_encoding = ','.join(word_encoding)
    entity_encoding = ','.join(entity_encoding)
    return word_encoding, entity_encoding


def transform(input_file, output_file):
    reader = open(input_file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    for line in reader:
        array = line.strip().split('\t')
        user_id = array[0]
        title = array[1]
        label = array[2]
        entities = array[3]
        word_encoding, entity_encoding = encoding_title(title, entities)
        writer.write('%s\t%s\t%s\t%s\n' %
                     (user_id, word_encoding, entity_encoding, label))
    reader.close()
    writer.close()


if __name__ == '__main__':
    print('counting frequencies of words and entities ...')
    count_word_and_entity_freq('./data/news/raw_news.txt')

    print('saving entity2name ...')
    writer = open('./data/sub_kg/entity2name.txt', 'w', encoding='utf-8')
    for k, v in dict(sorted(entity2name.items())).items():
        writer.write('%d\t%s\n' % (k, v))
    writer.close()

    print('saving triple2name')
    triple2name_writer = open('./data/sub_kg/triple2name.txt',
                              'w',
                              encoding='utf-8')
    triple2id_writer = open('./data/sub_kg/triple2id.txt',
                            'w',
                            encoding='utf-8')
    triple = pd.read_table('data/kg/train2id.txt', skiprows=[0], header=None)
    for x in triple.itertuples(index=False):
        if x[0] in entity2name and x[1] in entity2name:
            triple2name_writer.write(
                '%s\t%s\t%d\n' % (entity2name[x[0]], entity2name[x[1]], x[2]))
            triple2id_writer.write('%d\t%d\t%d\n' % (x[0], x[1], x[2]))

    triple2name_writer.close()
    triple2id_writer.close()

    print('constructing word2id map ...')
    construct_word2id()

    print('transforming training and test dataset ...')
    transform('./data/news/raw_news.txt', './data/news/news.txt')
