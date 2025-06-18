
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import numpy as np
from src.sentimentanalyzer.utils.common import load_saved_labels_and_texts
from sentimentanalyzer.entity import DataTransformationConfig
from sentimentanalyzer.logging import logger
from sentimentanalyzer.constants import *
from typing import List
import re
from src.sentimentanalyzer.utils.common import load_saved_labels_and_texts,preprocess_ft_txt



class DataTransformation:
    def __init__(self, config):
        self.config = config
        
        # Load preprocessed data: expects a dict with 'train' and 'test' keys,
        # each is a tuple: (labels_list, texts_list)
        self.data = load_saved_labels_and_texts(self.config.data_path)
        
        if not self.data or "train" not in self.data or "test" not in self.data:
            raise ValueError("❌ Data must contain 'train' and 'test' keys with labels and texts.")
        
        train_labels, train_texts = self.data.get("train")
        test_labels, test_texts = self.data.get("test")
        
        print("📊 Train:", len(train_labels), train_texts[:2])
        print("📊 Test:", len(test_labels), test_texts[:2])
        
        # Normalize texts right away
        self.train_texts = self.normalize_texts(train_texts)
        self.test_texts = self.normalize_texts(test_texts)
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        # Save normalized data back in fastText format
        self.save_to_ft_format(
            self.train_labels, self.train_texts, 
            os.path.join(self.config.root_dir, "train_ft.txt")
        )
        self.save_to_ft_format(
            self.test_labels, self.test_texts, 
            os.path.join(self.config.root_dir, "test_ft.txt")
        )
    
    def normalize_texts(self, texts: List[str]) -> List[str]:
        NON_ALPHANUM = re.compile(r'[\W]')
        NON_ASCII = re.compile(r'[^a-z0-1\s]')
        normalized_texts = []
        
        for text in texts:
            lower = text.lower()
            no_punctuation = NON_ALPHANUM.sub(' ', lower)
            no_non_ascii = NON_ASCII.sub('', no_punctuation)
            normalized_texts.append(no_non_ascii.strip())
        
        return normalized_texts
    
    def save_to_ft_format(self, labels: List[str], texts: List[str], filepath: str):
        assert len(labels) == len(texts), "❌ Labels and texts length mismatch"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for label, text in zip(labels, texts):
                f.write(f"__label__{label} {text}\n")
        
        print(f"💾 Saved: {filepath}")

