import gensim.downloader
import numpy as np
import pandas as pd
import re
import spacy

from sklearn.preprocessing import LabelBinarizer

SEED = 7

nlp = spacy.load('en_core_web_sm')


def load_data(dataset_name='train'):
    """
    Load data for the selected dataset.

    Label defintions
        (a) No Risk (or “None”): I don’t see evidence that this person is at risk for suicide.
        (b) Low Risk: There may be some factors here that could suggest risk, but I don’t really think this person is at much of a risk of suicide.
        (c) Moderate Risk: I see indications that there could be a genuine risk of this person making a suicide attempt.
        (d) Severe Risk: I believe this person is at high risk of attempting suicide in the near future.
        None: controls, which are not assigned a value for this variable.
    """
    df_posts = pd.DataFrame()
    df_labels = pd.DataFrame()
    df_users = pd.DataFrame()

    if dataset_name == 'train':
        df_users = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/train/task_A_train.posts.csv')
        df_posts = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv')
        df_labels = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/train/crowd_train.csv')
    if dataset_name == 'test':
        df_users = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/test/task_A_test.posts.csv')
        df_posts = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test.csv')
        df_labels = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/test/crowd_test_A.csv')

    df_posts = df_posts.loc[df_posts.user_id.isin(df_users.user_id)]
    df = df_users.merge(df_posts, on=['post_id', 'user_id', 'subreddit'])
    df = df.merge(df_labels, on='user_id').sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
    df.post_title.fillna(value='', inplace=True)
    df.post_body.fillna(value='', inplace=True)

    return df


def merge_texts(df):
    # merge titles and bodies into a single document for each post
    post_titles = []
    for doc in nlp.pipe(df.post_title):
        post_titles.append(spacy_tokenize(doc))

    post_bodies = []
    for doc in nlp.pipe(df.post_body):
        post_bodies.append(spacy_tokenize(doc))

    return [title + body for (title, body) in zip(post_titles, post_bodies)]


def create_token_index_mappings(texts):
    # create mappings of words to indices and indices to words
    UNK = '<UNKNOWN>'
    # PAD = '<PAD>'
    token_counts = {}

    for doc in texts:
        for token in doc:
            c = token_counts.get(token, 0) + 1
            token_counts[token] = c

    vocab = sorted(token_counts.keys())
    # start indexing at 1 as 0 is reserved for padding
    token2index = dict(zip(vocab, list(range(1, len(vocab) + 1))))
    token2index[UNK] = len(vocab) + 1
    # token2index[PAD] = len(vocab) + 2
    index2token = {value: key for (key, value) in token2index.items()}
    assert index2token[token2index['help']] == 'help'

    return token_counts, index2token, token2index


def prepare_sequential():
    df_train = load_data(dataset_name='train')
    df_test = load_data(dataset_name='test')

    df_train = df_train[['user_id', 'post_title', 'post_body', 'label']]
    df_test = df_test[['user_id', 'post_title', 'post_body', 'label']]

    texts_train = merge_texts(df_train)
    texts_test = merge_texts(df_test)

    # download GloVe embeddings
    glove_vectors = gensim.downloader.load('glove-twitter-200')

    token_counts_train, index2token_train, token2index_train = create_token_index_mappings(texts_train)

    # create mapping of words to their embeddings
    emb_map = {}
    for w in glove_vectors.vocab:
        emb_map[w] = glove_vectors.get_vector(w)

    vocab_size = len(token_counts_train)
    embed_len = glove_vectors['help'].shape[0]
    embedding_matrix = np.zeros((vocab_size + 1, embed_len))

    # initialize the embedding matrix
    for word, i in token2index_train.items():
        if i >= vocab_size:
            continue
        if word in glove_vectors:
            embedding_vector = glove_vectors.get_vector(word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    lb = LabelBinarizer()
    lb.fit(df_train.label)
    y_train = lb.transform(df_train.label)
    x_train = texts_train

    lb.fit(df_test.label)
    y_test = lb.transform(df_test.label)
    x_test = texts_test

    return x_train, y_train, x_test, y_test, embedding_matrix


def spacy_tokenize(doc):
    if isinstance(doc, str):
        doc = nlp(doc)
    tokens = []
    for token in doc:
        if token.is_punct:
            continue
        elif token.is_space:
            continue
        elif token.like_url:
            tokens.append('__URL__')
        elif token.like_num:
            tokens.append('__NUM__')
        elif re.search('.+_person_.+', token.lower_) is not None:
            split = [token for token in token.lower_.split('_person_') if token != '']
            tokens.extend(split)
        elif ',' in token.lower_:
            split = [token for token in token.lower_.split(',') if token != '']
            tokens.extend(split)
        else:
            form = token.lower_
            form = re.sub('[\!\"#\$%&\(\)\*\+,\./:;<=>\?@\[\\]\^_`\{\|\}\~]+', '', form)
            form = re.sub('([^\-,]+)[\-,]', '\g<1>', form)
            form = re.sub('^([^\.]+)\.', '\g<1>', form)
            tokens.append(form)
    return tokens




