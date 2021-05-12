from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D
from keras.optimizers import Adam
from utils import prepare_elmo, MAX_LENGTH


class ElmoCNN(Sequential):
    def __init__(self):
        super().__init__()

    def build_model(self, optimizer=Adam(lr=0.001), loss='categorical_crossentropy'):
        self.add(Conv1D(1, 400, activation='relu', padding='same'))
        self.add(MaxPooling1D())
        self.add(Flatten())
        self.add(Dropout(0.5))
        self.add(Dense(4, activation='softmax'))
        self.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.summary()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = prepare_elmo()

    elmo_cnn = ElmoCNN()
    elmo_cnn.build_model()
    history = elmo_cnn.fit()