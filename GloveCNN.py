from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from utils import prepare_sequential, MAX_LENGTH
from simple_elmo import ElmoModel


class GloveCNN(Sequential):
    def __init__(self, emb_matrix, emb_len, optimizer, loss):
        super().__init__()
        self.emb_matrix = emb_matrix
        self.emb_len = emb_len
        self.optimizer = optimizer
        self.loss = loss

    def build_model(self, lr=0.001, metrics=['accuracy']):
        self.add(Embedding(input_dim=self.emb_matrix.shape[0], output_dim=self.embed_len, input_length=MAX_LENGTH,
                           weights=[self.emb_matrix], trainable=False))
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
        self.compile(loss=self.loss, optimizer=self.optimizer, lr=lr, metrics=metrics)
        self.summary()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, emb_matrix = prepare_sequential(merge=False, emb_name='glove-wiki-gigaword-300')

    glove_cnn = GloveCNN(emb_matrix, emb_matrix[0].shape[0], 'adam', 'categorical_crossentropy')
