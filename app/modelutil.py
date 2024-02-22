import os 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same', name='conv3d_1'))
    model.add(Activation('relu', name='activation_1'))
    model.add(MaxPool3D((1,2,2), name='max_pooling3d_1'))

    model.add(Conv3D(256, 3, padding='same', name='conv3d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPool3D((1,2,2), name='max_pooling3d_2'))

    model.add(Conv3D(75, 3, padding='same', name='conv3d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(MaxPool3D((1,2,2), name='max_pooling3d_3'))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax', name='dense_output'))

    model.load_weights(os.path.join('..','models','checkpoint'))

    return model


