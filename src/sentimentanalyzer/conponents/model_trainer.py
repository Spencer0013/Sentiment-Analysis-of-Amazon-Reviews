import ast
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from sentimentanalyzer.entity import ModelTrainerConfig
from src.sentimentanalyzer.utils.common import load_fasttext_file, convert_labels
from sklearn.model_selection import train_test_split
from sentimentanalyzer.logging import logger
import pandas as pd
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense,Dropout
from tensorflow.keras.models import Sequential

tf.random.set_seed(42)


class ModelTrainer:
    def __init__(self, config: 'ModelTrainerConfig'):
        self.config = config
        self.params = config
        tf.random.set_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        train_path = Path(config.data_path) / "train_ft.txt"
        test_path = Path(config.data_path) / "test_ft.txt"

        self.train_texts, self.train_labels = load_fasttext_file(train_path)
        self.test_texts, self.test_labels = load_fasttext_file(test_path)

        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
        self.train_texts, self.train_labels, random_state=42, test_size=0.2)

        self.train_texts = self.train_texts
        self.train_labels = self.train_labels
        self.val_texts = self.val_texts
        self.val_labels =self. val_labels

        self.test_texts = self.test_texts
        self.test_labels = self.test_labels

        self.train_labels = convert_labels(self.train_labels)
        self.val_labels = convert_labels(self.val_labels)
        self.test_labels = convert_labels(self.test_labels)



        
    def train(self):
        inputs = tf.keras.Input(
            shape=(),    
            dtype=tf.string
        )

        vectorizer = layers.TextVectorization(
            max_tokens=self.config.max_tokens,
            output_sequence_length=self.config.output_sequence_length,
            standardize="lower_and_strip_punctuation",
            split="whitespace"
        )
        vectorizer.adapt(self.train_texts)
        x = vectorizer(inputs)
        x = layers.Embedding(
            input_dim=self.config.max_tokens,
            output_dim=self.config.output_dim,
            mask_zero=True
        )(x)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(3)(x)
        x = layers.Conv1D(64, 5, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)
        x = layers.Conv1D(64, 5, activation='relu')(x)
        x = layers.GlobalMaxPool1D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(20,activation='relu')(x)
        # 4) Classification head — use `classes` from params
        outputs = layers.Dense(
            units=self.params.classes,
            activation="softmax",
            name="classifier"
        )(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="EmbeddingConv1DModel")

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.params.learning_rate),
            loss=losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"]
        )
        model.summary()

        # 7) Train
        history = model.fit(
            x=self.train_texts,
            y=self.train_labels,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(
                self.val_texts,
                self.val_labels
            )
        )
        # 8) Save
        model.save(self.config.model_save_path, save_format='tf')
        logger.info(f"Model saved to {self.config.model_save_path}")


     

        return model
