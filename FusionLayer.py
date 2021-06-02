import keras
import numpy as np
from keras.layers import Conv1D, Dense, Dropout, Embedding, MaxPooling1D, AveragePooling1D, SimpleRNN, Input, \
    Bidirectional, GRU, LSTM, concatenate
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from utils import prepare_sequential, MAX_LENGTH
from sklearn.utils import class_weight


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, emb_matrix = prepare_sequential(merge=False, emb_name='glove-wiki-gigaword-300')

    inputs = Input(shape=(emb_matrix.shape[0]))
    e = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix[0].shape[0], input_length=MAX_LENGTH,
                  weights=[emb_matrix], trainable=False)(inputs)

    # GloveCNN
    gcnn_c1 = Conv1D(2, 300, activation='relu', padding='same')(e)
    gcnn_do1 = Dropout(0.5)(gcnn_c1)
    gcnn_mp2 = MaxPooling1D()(gcnn_do1)
    gcnn_ap = AveragePooling1D()(gcnn_do1)
    gcnn_out = concatenate([gcnn_mp2, gcnn_ap])

    # GloveBiRNN
    grnn_r1 = Bidirectional(SimpleRNN(64, return_sequences=True))(e)
    grnn_do1 = Dropout(0.5)(grnn_r1)
    grnn_r2 = Bidirectional(SimpleRNN(64, return_sequences=True))(grnn_do1)
    grnn_do2 = Dropout(0.5)(grnn_r2)
    grnn_mp = MaxPooling1D()(grnn_do2)
    grnn_ap = AveragePooling1D()(grnn_do2)
    grnn_out = concatenate([grnn_mp, grnn_ap])

    # GloveGRU
    ggru_r1 = GRU(64, return_sequences=True)(e)
    ggru_do1 = Dropout(0.5)(ggru_r1)
    ggru_r2 = GRU(64, return_sequences=True)(ggru_do1)
    ggru_do2 = Dropout(0.5)(ggru_r2)
    ggru_mp = MaxPooling1D()(ggru_do2)
    ggru_ap = AveragePooling1D()(ggru_do2)
    ggru_out = concatenate([ggru_mp, ggru_ap])

    # GloveBiLSTM
    glstm_r1 = Bidirectional(LSTM(32, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                  return_sequences=True))(e)
    glstm_do1 = Dropout(0.5)(glstm_r1)
    glstm_r2 = Bidirectional(LSTM(32, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                  return_sequences=True))(glstm_do1)
    glstm_do2 = Dropout(0.5)(glstm_r2)
    glstm_mp = MaxPooling1D()(glstm_do2)
    glstm_ap = AveragePooling1D()(glstm_do2)
    glstm_out = concatenate([glstm_mp, glstm_ap])

    output_1 = concatenate([gcnn_out, grnn_out, ggru_out, glstm_out])
    outputs = Dense(4)(output_1)

    model = keras.Model(inputs=inputs, outputs=outputs, name='fusion_model')
    model.summary()

    class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced', np.unique(y_train),
                                                                                   y_train)))
    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(lr=10**-4), loss_weights=class_weights)
