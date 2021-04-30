from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from utils import prepare_sequential


MAX_LENGTH = 400


X_train, y_train, X_test, y_test, embedding_matrix = prepare_sequential()


def build_model(n_units=32, fc_dim=32, lr=0.001):
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

    return model


lstm_model = KerasClassifier(build_fn=build_model, n_units=64, fc_dim=256, lr=0.00001, verbose=1)