
# import tensorflow as tf
# import pandas as pd
# import tensorflow_hub as hub
# from tensorflow.keras import layers, models, losses, optimizers
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sentimentanalyzer.entity import ModelTrainerUSEConfig
# tf.random.set_seed(42)
# from pathlib import Path
# import os
# from tensorflow.keras.layers import Dense



# class ModelTrainerUSE:
#     def __init__(self, config):
#         self.config = config
#         self.params = config
#         self.batch_size = 32
#         self.shuffle = True
#         self.model = None

#     def build_model(self):
#         inputs = tf.keras.Input(shape=(), dtype=tf.string, name='text_input')
    
#     # Universal Sentence Encoder (frozen)
#         use_layer = hub.KerasLayer(
#         self.config.use_model_path,
#         input_shape=[],
#         dtype=tf.string,
#         trainable=True,  # Keep frozen for efficiency
#         name='USE_encoder'
#         )(inputs)
    
#     # Batch Normalization
#         x = tf.keras.layers.BatchNormalization()(use_layer)
    
#     # Dense layers with regularization
#         x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
#         x = tf.keras.layers.Dropout(0.3)(x)
#         x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
#         x = tf.keras.layers.Dropout(0.3)(x)
    
#     # Output layer (no softmax activation for stability)
#         outputs = tf.keras.layers.Dense(self.config.classes)(x)
    
#     # Build model
#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         self.model.summary()
    
#     # Compile with improved settings
#         self.model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=self.config.learning_rate,
#             clipnorm=1.0  # Gradient clipping
#         ),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
#     )

#     def load_data(self):
#         data_path = Path(self.config.data_path)
#         df_train = pd.read_csv(data_path / 'train_clean.csv')[:50000]
#         df_test = pd.read_csv(data_path / 'test_clean.csv')[:50000]

#         train_df, valid_df = train_test_split(df_train, test_size=0.2, random_state=42)

#         self.train_df = train_df
#         self.valid_df = valid_df
#         self.df_test = df_test


#     def load_and_encode_data(self):
#         le = LabelEncoder().fit(self.train_df['target'])
#         # Transform and replace in-place by assignment
#         self.train_df['target'] = le.transform(self.train_df['target'])
#         self.valid_df['target'] = le.transform(self.valid_df['target'])
#         self.df_test['target'] = le.transform(self.df_test['target'])
    
#     def df_to_tf_dataset(self, df, shuffle=True, batch_size=32):
#         texts = df['text'].astype(str).tolist()
#         labels = df['target'].tolist()  # or 'label' depending on your df
#         ds = tf.data.Dataset.from_tensor_slices((texts, labels))
#         if shuffle:
#             ds = ds.shuffle(buffer_size=len(df))
#         return ds.batch(batch_size)

#     def prepare_datasets(self):
#         self.train_ds = self.df_to_tf_dataset(self.train_df)
#         self.valid_ds = self.df_to_tf_dataset(self.valid_df, shuffle=False)
#         self.test_ds = self.df_to_tf_dataset(self.df_test, shuffle=False)

#     def train(self):
#         self.model.fit(
#             self.train_ds,
#             validation_data=self.valid_ds,
#             epochs=self.config.epochs
#         )
#         self.model.save(self.config.model_save_path)
#         self.model.summary()


#     def save_tf_datasets(self, save_dir=None):

#       """
#       1) prepare train/valid/test tf.data.Dataset (batched)
#       2) serialize each split to TFRecord under save_dir
#       """
#       save_dir = self.config.root_dir
#       os.makedirs(save_dir, exist_ok=True)

#       self.prepare_datasets()

#       for split, ds in (('train', self.train_ds),
#                       ('valid', self.valid_ds),
#                       ('test',  self.test_ds)):
#           path = os.path.join(save_dir, f"{split}.tfrecord")
#           with tf.io.TFRecordWriter(path) as writer:
#               for text_batch, target_batch in ds:
#                   for t, tgt in zip(text_batch, target_batch):
#                       ex = tf.train.Example(features=tf.train.Features(feature={
#                           'text':   tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.numpy()])),
#                           'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(tgt.numpy())]))
#                       }))
#                       writer.write(ex.SerializeToString())

#       print(f"✔️  TFRecords written to {save_dir}: "
#              f"train.tfrecord, valid.tfrecord, test.tfrecord")