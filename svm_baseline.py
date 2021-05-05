from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from utils import load_data, spacy_tokenize

SEED = 7

if __name__ == '__main__':
    df_train = load_data(dataset_name='train')
    df_train = df_train[['user_id', 'post_title', 'post_body', 'label']]

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer=spacy_tokenize)),
        ('svm', LinearSVC(class_weight='balanced'))
    ])

    # prepare the data
    X = [title + body for (title, body) in zip(df_train.post_title.tolist(), df_train.post_body.tolist())]
    y = df_train.label.to_list()

    le = LabelEncoder()
    y_ = le.fit_transform(y)

    X_train, X_dev, y_train, y_dev = train_test_split(X, y_, test_size=0.2, random_state=SEED)

    # default configuration with spaCy tokenizer (lowercased words) gives best macro-averaged results
    # params = {'svm__C': [0.001, 0.01, 1.0, 10, 100], 'svm__class_weight': ['balanced', None]}
    # gs = GridSearchCV(pipeline, param_grid=params, cv=10, scoring='accuracy')
    # gs.fit(Xb_train, yb_train)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_dev)
    print(classification_report(y_dev, y_pred, target_names=le.classes_))
