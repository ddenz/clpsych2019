from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D, SimpleRNN, Bidirectional
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from utils import prepare_sequential, MAX_LENGTH
from simple_elmo import ElmoModel


class GloveCNN(Sequential):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.emb_matrix = embedding_matrix

    def build_model_(self, lr=0.001, optimizer='adam', loss='categorical_crossentropy'):
        self.add(Embedding(input_dim=self.emb_matrix.shape[0], output_dim=self.emb_matrix[0].shape[0],
                           input_length=MAX_LENGTH, weights=[self.emb_matrix], trainable=False))
        self.add(Conv1D(64, 7, activation='relu', padding='same'))
        self.add(MaxPooling1D())
        self.add(Conv1D(128, 5, activation='relu', padding='same'))
        self.add(MaxPooling1D())
        self.add(Conv1D(256, 3, activation='relu', padding='same'))
        self.add(MaxPooling1D())
        self.add(Flatten())
        self.add(Dense(64, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4, activation='softmax'))
        self.compile(loss=loss, optimizer=optimizer, lr=lr, metrics=['accuracy'])
        self.summary()

    def build_model(self, optimizer=Adam(lr=0.001), loss='categorical_crossentropy'):
        self.add(Embedding(input_dim=self.emb_matrix.shape[0], output_dim=self.emb_matrix[0].shape[0],
                           input_length=MAX_LENGTH, weights=[self.emb_matrix], trainable=False))
        self.add(Conv1D(2, 300, activation='relu', padding='same'))
        self.add(MaxPooling1D())
        self.add(Flatten())
        self.add(Dropout(0.5))
        self.add(Dense(4, activation='softmax'))
        self.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.summary()


class GloveBiRNN(Sequential):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.emb_matrix = embedding_matrix

    def build_model(self, optimizer=Adam(lr=0.001), loss='categorical_crossentropy'):
        self.add(Embedding(input_dim=self.emb_matrix.shape[0], output_dim=self.emb_matrix[0].shape[0],
                           input_length=MAX_LENGTH, weights=[self.emb_matrix], trainable=False))
        self.add(Bidirectional(SimpleRNN(64, return_sequences=True)))
        self.add(Dropout(0.5))
        self.add(Bidirectional(SimpleRNN(64)))
        self.add(Dropout(0.5))
        self.add(Dense(4, activation='softmax'))
        self.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.summary()


if __name__ == '__main__':
    #X_train, y_train, X_test, y_test, emb_matrix = prepare_sequential(merge=False, emb_name='glove-wiki-gigaword-300')

    #glove_cnn = GloveCNN(emb_matrix)
    #glove_cnn.build_model()
    #history = glove_cnn.fit(X_train, y_train, batch_size=32, epochs=20)

    X_train, y_train, X_test, y_test, emb_matrix = prepare_sequential(merge=True)
    print('shape=', emb_matrix.shape)
    glove_rnn = GloveBiRNN(emb_matrix)
    glove_rnn.build_model()
    history = glove_rnn.fit(X_train, y_train, batch_size=32, epochs=20)
