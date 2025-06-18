
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
from sentimentanalyzer.utils.common import save_json, load_fasttext_file,convert_labels
from pathlib import Path
from sentimentanalyzer.entity import EvaluationConfig
import os

class Evaluation:
    def __init__(self, config: 'EvaluationConfig'):
        self.config = config
        self.model = None
        self.test_ds = None
        self.score = None
       
        test_path = Path(config.data_path) / "test_ft.txt"
        self.test_texts, self.test_labels = load_fasttext_file(test_path)
        self.test_texts = self.test_texts
        self.test_labels = self.test_labels

        self.test_labels = convert_labels(self.test_labels)

    def load_model(self) -> tf.keras.Model:
        """
        Load and return the Keras model from the configured path.
        """
        model_path: Path = self.config.path_of_model
        self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        return self.model

    def evaluate(self) -> tuple:
         self.load_model()
         self.score = self.model.evaluate(
             x =self.test_texts,
             y =self.test_labels
         )
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
