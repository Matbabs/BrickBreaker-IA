import tensorflow as tf

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