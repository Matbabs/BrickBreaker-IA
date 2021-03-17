import tensorflow as tf
import random

class NeuralNetwork():
    def __init__(self, path=None):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(3,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    def train(self, values, targets):
        self.model.fit(values, targets, epochs=5)

    def predict(self, values):
        return self.model.predict(values)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def randomize_weights(self, magnitude):
        for layer in range(2,7):
            new_weights = self.model.layers[layer].get_weights()
            for weights in range(len(new_weights[0])):
                for weight in range(len(new_weights[0][weights])):
                    new_weights[0][weights][weight] += random.randrange(-9,9)/magnitude
            self.model.layers[layer].set_weights(new_weights)