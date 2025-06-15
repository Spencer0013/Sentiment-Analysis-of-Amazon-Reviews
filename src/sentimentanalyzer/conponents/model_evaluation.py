
import tensorflow as tf
import json
from pathlib import Path
from sentimentanalyzer.utils.common import save_json
from sentimentanalyzer.entity import EvaluationConfig
import tensorflow_hub as hub
import os

class Evaluation:
    def __init__(self, config: 'EvaluationConfig'):
        self.config = config
        self.model = None
        self.test_ds = None
        self.score = None
       

    def load_model(self) -> tf.keras.Model:
        """
        Load and return the Keras model from the configured path.
        """
        model_path: Path = self.config.path_of_model
        self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        return self.model


    def load_test_dataset(self, batch_size: int = 32, tfrecord_path: str = None):
       
       """
       Load the test split from TFRecord—deterministic
       """
       # Determine file path
       path = tfrecord_path or os.path.join(self.config.test_data)
   
       # Define feature spec once
       feature_spec = {
        'text':   tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
       }

       # Parser function (no lambda)
       def _parse_example(serialized):
           parsed = tf.io.parse_single_example(serialized, feature_spec)
           return parsed['text'], parsed['target']

       # Build dataset
       ds = (
            tf.data.TFRecordDataset(path)
            .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

       self.test_ds = ds
       return ds



    def evaluate(self) -> tuple:
         self.load_model()
         self.test_ds = self.load_test_dataset()
         self.score = self.model.evaluate(self.test_ds)
         self.save_score()
         return self.score

       

    def save_score(self, output_path: Path = Path("scores.json")) -> None:
        """
        Save evaluation results to a JSON file at output_path.
        """
        if self.score is None:
            raise ValueError("No evaluation score to save. Run evaluate() first.")

        loss, accuracy = self.score
        scores = {"loss": float(loss), "accuracy": float(accuracy)}
        save_json(path=output_path, data=scores)
