import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation, ReLU, Dropout, MaxPooling2D, Conv2D
from keras.layers import Flatten, Input
from keras.regularizers import l2


def build_ann_model() :
    return Sequential(
        [
            Dense(32, input_shape=(784,), kernel_initializer='he_uniform',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            ReLU(),
            Dropout(0.1),
            Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            ReLU(),
            Dropout(0.1),
            Dense(10, activation='softmax')
        ]
    )

def build_cnn_model():
    return Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ]
    )


def build_cnn2_model():
    return Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            Conv2D(32, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            Conv2D(32, 5, padding='valid', strides=2),
            BatchNormalization(),
            ReLU(),
            Dropout(0.4),

            Conv2D(64, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 3, padding='valid', strides=1, ),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 5, padding='valid', strides=2, ),
            BatchNormalization(),
            ReLU(),
            Dropout(0.4),

            Flatten(),
            Dense(128, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            ReLU(),
            Dropout(0.4),

            Dense(10, activation='softmax')
        ]
    )


