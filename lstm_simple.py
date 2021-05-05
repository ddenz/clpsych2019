import logging
import numpy as np

from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import classification_report
from utils import prepare_sequential


MAX_LENGTH = 400

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def build_model(n_units=32, fc_dim=32, lr=0.001):
    logging.info('Initializing model...')
    model = Sequential()
    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix[0].shape[0],
                                input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(n_units, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                   return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(fc_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    logging.info(model.summary())
    return model


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, embedding_matrix = prepare_sequential()

    model = KerasClassifier(build_fn=build_model, n_units=64, fc_dim=256, lr=0.00001, verbose=1)
    history = model.fit(X_train, y_train, batch_size=32, epochs=20)
    y_pred = model.predict(X_test)

    y_pred_m = [np.argmax(y) for y in y_pred]
    y_test_m = [np.argmax(y) for y in y_test]

    print(classification_report(y_test_m, y_pred_m))
