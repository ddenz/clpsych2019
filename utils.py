import gensim.downloader
import logging
import numpy as np
import pandas as pd
import pickle
import re
import spacy

from keras.preprocessing.sequence import pad_sequences
from simple_elmo import ElmoModel
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import Sequence

SEED = 7
UNK = '<UNKNOWN>'
MAX_LENGTH = 400

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load('en_core_web_sm')


class DatasetGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size=4):
        self.x = x_set
        self.y = LabelBinarizer().fit_transform(y_set)

        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def load_data(dataset_name='train', reload=False):
    """
    Load data for the selected dataset.

    Label defintions
        (a) No Risk (or “None”): I don’t see evidence that this person is at risk for suicide.
        (b) Low Risk: There may be some factors here that could suggest risk, but I don’t really think this person is at much of a risk of suicide.
        (c) Moderate Risk: I see indications that there could be a genuine risk of this person making a suicide attempt.
        (d) Severe Risk: I believe this person is at high risk of attempting suicide in the near future.
        None: controls, which are not assigned a value for this variable.
    """
    logging.info('Loading dataset: ' + dataset_name)

    df_posts = pd.DataFrame()
    df_labels = pd.DataFrame()
    df_users = pd.DataFrame()

    if reload:
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
    else:
        df = pd.read_csv('umd_reddit_suicidewatch_dataset_v2/crowd/' + dataset_name + '/task_A_' + dataset_name + '.csv', keep_default_na=False)

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
    logging.info('Creating token-index mappings...')
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


def load_embeddings(emb_name):
    # download embeddings if not already available
    logging.info('Downloading embeddings: ' + emb_name)
    return gensim.downloader.load(emb_name)


def prepare_sequential(merge=False, emb_name='glove-twitter-200'):
    logging.info('Preparing sequential data (' + emb_name + ')...')

    df_train = load_data(dataset_name='train')
    df_test = load_data(dataset_name='test')

    df_train = df_train[['user_id', 'post_title', 'post_body', 'label']]
    df_test = df_test[['user_id', 'post_title', 'post_body', 'label']]

    if merge:
        texts_train = merge_texts(df_train)
        texts_test = merge_texts(df_test)
    else:
        texts_train = []
        for doc in nlp.pipe(df_train.post_body):
            texts_train.append(spacy_tokenize(doc))

        texts_test = []
        for doc in nlp.pipe(df_test.post_body):
            texts_test.append(spacy_tokenize(doc))

    embedding_vectors = load_embeddings(emb_name)

    token_counts, index2token, token2index = create_token_index_mappings(texts_train + texts_test)

    # create mapping of words to their embeddings
    emb_map = {}
    for w in embedding_vectors.vocab:
        emb_map[w] = embedding_vectors.get_vector(w)

    vocab_size = len(token_counts)
    embed_len = embedding_vectors['help'].shape[0]
    embedding_matrix = np.zeros((vocab_size + 1, embed_len))

    # initialize the embedding matrix
    logging.info('Initializing embeddings matrix...')
    for word, i in token2index.items():
        if i >= vocab_size:
            continue
        if word in embedding_vectors:
            embedding_vector = embedding_vectors.get_vector(word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    logging.info('Preparing train data...')
    lb = LabelBinarizer()
    lb.fit(df_train.label)
    y_train = lb.transform(df_train.label)
    x_train = [[token2index.get(token, token2index[UNK]) for token in doc] for doc in texts_train]
    x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding='post')

    logging.info('Preparing test data...')
    lb.fit(df_test.label)
    y_test = lb.transform(df_test.label)
    x_test = [[token2index.get(token, token2index[UNK]) for token in doc] for doc in texts_test]
    x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding='post')

    return x_train, y_train, x_test, y_test, embedding_matrix


def prepare_elmo(load_from_file=False):
    logging.info('Preparing sequential data (Elmo)...')

    if load_from_file:
        logging.info('Loading from file...')
        x_train = pickle.load(open('embeddings/X_train_elmo.pickle', 'rb'))
        y_train = pickle.load(open('embeddings/y_train_elmo.pickle', 'rb'))
        x_test = pickle.load(open('embeddings/X_test_elmo.pickle', 'rb'))
        y_test = pickle.load(open('embeddings/y_test_elmo.pickle', 'rb'))
        return x_train, y_train, x_test, y_test

    elmo_model = ElmoModel()
    elmo_model.load('embeddings/193.zip')

    df_train = load_data(dataset_name='train')
    df_test = load_data(dataset_name='test')

    df_train = df_train[['user_id', 'post_title', 'post_body', 'label']]
    df_test = df_test[['user_id', 'post_title', 'post_body', 'label']]

    #x_train = []
    texts_train = []
    for doc in nlp.pipe(df_train.post_body):
        #texts_train.append([spacy_tokenize(doc) for sent in doc.sents])
        texts_train.append(spacy_tokenize(doc))
        #x_train.append(elmo_model.get_elmo_vector_average(spacy_tokenize(doc)))

    #x_test = []
    texts_test = []
    for doc in nlp.pipe(df_test.post_body):
        #texts_test.append([spacy_tokenize(sent) for sent in doc.sents])
        texts_test.append(spacy_tokenize(doc))
        #x_test.append(elmo_model.get_elmo_vector_average(spacy_tokenize(doc)))

    print('x_train:', np.asarray(texts_train).shape)
    print('x_test :', np.asarray(texts_test).shape)

    x_train = elmo_model.get_elmo_vector_average(texts_train)
    x_test = elmo_model.get_elmo_vector_average(texts_test)

    print('x_train.shape:' + str(x_train.shape))
    print('x_test.shape :' + str(x_test.shape))

    logging.info('Preparing train data...')
    lb = LabelBinarizer()
    lb.fit(df_train.label)
    y_train = lb.transform(df_train.label)

    logging.info('Preparing test data...')
    lb.fit(df_test.label)
    y_test = lb.transform(df_test.label)

    logging.info('Saving data to files...')
    pickle.dump(x_train, open('embeddings/X_train_elmo.pickle', 'wb'))
    pickle.dump(y_train, open('embeddings/y_train_elmo.pickle', 'wb'))
    pickle.dump(x_test, open('embeddings/X_test_elmo.pickle', 'wb'))
    pickle.dump(y_test, open('embeddings/y_test_elmo.pickle', 'wb'))

    return x_train, y_train, x_test, y_test


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
