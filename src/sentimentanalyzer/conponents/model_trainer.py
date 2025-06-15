import ast
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from sentimentanalyzer.entity import ModelTrainerConfig
tf.random.set_seed(42)


class ModelTrainer:
    def __init__(self, config: 'ModelTrainerConfig'):
        self.config = config
        self.params = config.params
        # (You already loaded these embeddings earlier if needed)
        self.token_embed = np.load("artifacts/data_transformation/token_embeddings.npy")

    def train(self):
        # 1) Ensure input_shape is a tuple
        input_shape = self.config.input_shape
        if isinstance(input_shape, str):
            input_shape = tuple(ast.literal_eval(input_shape))

        # 2) Define Input layer for precomputed float32 embeddings
        inputs = layers.Input(
            shape=input_shape,    # e.g. (163, 107)
            dtype=tf.float32,     # embeddings are float32
            name="embedding_input"
        )

        # 3) Conv + pooling
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
        x = layers.GlobalAveragePooling1D()(x)

        # 4) Classification head — use `classes` from params
        outputs = layers.Dense(
            units=self.params.classes,
            activation="softmax",
            name="classifier"
        )(x)

        # 5) Build & compile
        model = models.Model(inputs=inputs, outputs=outputs, name="EmbeddingConv1DModel")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.params.learning_rate),
            loss=losses.SparseCategoricalCrossentropy(),  # switch here
            metrics=["accuracy"]
        )

    
        
        model.summary()

        # 6) Load tf.data datasets (must yield (embeddings, labels))
        train_ds = tf.data.experimental.load("artifacts/data_transformation/train_ds")
        valid_ds = tf.data.experimental.load("artifacts/data_transformation/valid_ds")

        # 7) Train
        model.fit(
            train_ds,
            epochs=self.params.epochs,
            validation_data=valid_ds
        )

        # 8) Save
        model.save("artifacts/model_trainer/model.h5")

        return model
