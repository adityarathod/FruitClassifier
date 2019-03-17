import tensorflow as tf
from tensorflow import keras


def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=(5, 5), input_shape=(100,100,4)),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Conv2D(32, kernel_size=(5, 5)),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Conv2D(64, kernel_size=(5, 5)),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Conv2D(128, kernel_size=(5, 5)),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Dense(1024),
        keras.layers.Dense(256),
        keras.layers.Softmax()
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
