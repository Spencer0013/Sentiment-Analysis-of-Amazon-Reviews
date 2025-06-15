
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sentimentanalyzer.logging import logger
from sentimentanalyzer.constants import *
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import numpy as np
from sentimentanalyzer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config):
        self.config = config


        self.vectorizer = TextVectorization(
            max_tokens=config.max_tokens,
            output_sequence_length=config.output_sequence_length,
            standardize="lower_and_strip_punctuation",
            split="whitespace"
        )
        self.embedding = Embedding(
            input_dim=config.max_tokens,
            output_dim=config.output_dim,
            mask_zero=True
        )

    def transform(self):
        # Your token-embedding based transformation pipeline
        df_train = pd.read_csv(self.config.data_path_train)
        df_test = pd.read_csv(self.config.data_path_test)

        train_df, valid_df = train_test_split(df_train, test_size=0.2, random_state=42)
        X_train, X_valid = train_df['text'].astype(str), valid_df['text'].astype(str)
        X_test = df_test['text'].astype(str)

        le = LabelEncoder().fit(train_df['target'])
        y_train, y_valid = le.transform(train_df['target']), le.transform(valid_df['target'])
        y_test = le.transform(df_test['target'])

        # Tokenize & embed
        self.vectorizer.adapt(X_train)
        t_train = self.embedding(self.vectorizer(tf.constant(X_train)))
        t_valid = self.embedding(self.vectorizer(tf.constant(X_valid)))
        t_test = self.embedding(self.vectorizer(tf.constant(X_test)))

        np.save(self.config.transformed_token_embedding_path, t_train.numpy())

        def _ds(X, y):
            return (tf.data.Dataset.from_tensor_slices((X, y))
                    .shuffle(1_000).batch(self.config.batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        train_ds = _ds(t_train, y_train)
        valid_ds = _ds(t_valid, y_valid)
        test_ds = _ds(t_test, y_test)

        os.makedirs(self.config.root_dir, exist_ok=True)
        train_path = os.path.join(self.config.root_dir, "train_ds")
        valid_path = os.path.join(self.config.root_dir, "valid_ds")
        test_path = os.path.join(self.config.root_dir, "test_ds")

        tf.data.experimental.save(train_ds, train_path)
        tf.data.experimental.save(valid_ds, valid_path)
        tf.data.experimental.save(test_ds, test_path)

        logger.info(f"Saved train_ds at {train_path}")
        logger.info(f"Saved valid_ds at {valid_path}")
        logger.info(f"Saved test_ds at {test_path}")
