from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D, AveragePooling1D, SimpleRNN, \
    Bidirectional, GRU, LSTM
from keras.models import Sequential
from keras.layers import Input, concatenate
from tensorflow.keras.layers import Attention
from utils import prepare_sequential, MAX_LENGTH


class FusionLayer(Sequential):
    pass


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, emb_matrix = prepare_sequential(merge=False, emb_name='glove-wiki-gigaword-300')

    print('input_dim=', emb_matrix.shape[0])
    print('output_dim=', emb_matrix[0].shape[0])

    inputs = Input(shape=(emb_matrix.shape[0]))
    e = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix[0].shape[0], input_length=MAX_LENGTH,
                  weights=[emb_matrix], trainable=False)(inputs)

    # GloveCNN
    gcnn_c1 = Conv1D(2, 300, activation='relu', padding='same')(e)
    gcnn_mp1 = MaxPooling1D()(gcnn_c1)
    gcnn_f1 = Flatten()(gcnn_mp1)
    gcnn_do1 = Dropout(0.5)(gcnn_f1)
    gcnn_d1 = Dense(4, activation='softmax')(gcnn_do1)
    gcnn_mp = MaxPooling1D()(gcnn_d1)
    gcnn_ap = AveragePooling1D()(gcnn_d1)
    gcnn_att = Attention(gcnn_d1)
    gcnn_out = concatenate([gcnn_mp, gcnn_ap, gcnn_att])

    # GloveBiRNN
    grnn_r1 = Bidirectional(SimpleRNN(64, return_sequences=True))(e)
    grnn_do1 = Dropout(0.5)(grnn_r1)
    grnn_r2 = Bidirectional(SimpleRNN(64, return_sequences=True))(grnn_do1)
    grnn_do2 = Dropout(0.5)(grnn_r2)
    grnn_d1 = Dense(4, activation='softmax')(grnn_do2)
    grnn_mp = MaxPooling1D()(grnn_d1)
    grnn_ap = AveragePooling1D()(grnn_d1)
    grnn_att = Attention(grnn_d1)
    grnn_out = concatenate([grnn_mp, grnn_ap, grnn_att])

    # GloveGRU
    ggru_r1 = GRU(64, return_sequences=True)(e)
    ggru_do1 = Dropout(0.5)(ggru_r1)
    ggru_r2 = GRU(64, return_sequences=False)(ggru_do1)
    ggru_do2 = Dropout(0.5)(ggru_r2)
    ggru_d1 = Dense(4, activation='softmax')(ggru_do2)
    ggru_mp = MaxPooling1D()(ggru_d1)
    ggru_ap = AveragePooling1D()(ggru_d1)
    ggru_att = Attention(ggru_d1)
    ggru_out = concatenate([ggru_mp, ggru_ap, ggru_att])

    # GloveBiLSTM
    glstm_r1 = Bidirectional(LSTM(32, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                  return_sequences=True))(e)
    glstm_do1 = Dropout(0.5)(glstm_r1)
    glstm_r2 = Bidirectional(LSTM(32, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                  return_sequences=False))(glstm_do1)
    glstm_do2 = Dropout(0.5)(glstm_r2)
    glstm_d1 = Dense(4, activation='softmax')(glstm_do2)
    glstm_mp = MaxPooling1D()(glstm_d1)
    glstm_ap = AveragePooling1D()(glstm_d1)
    glstm_att = Attention(glstm_d1)
    glstm_out = concatenate([grnn_mp, grnn_ap, grnn_att])

    x = concatenate([])