import pandas as pd
import re
import spacy


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


def create_token_index_mappings(all_text):
    # create mappings of words to indices and indices to words
    UNK = '<UNKNOWN>'
    # PAD = '<PAD>'
    token_counts = {}

    for doc in all_text:
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