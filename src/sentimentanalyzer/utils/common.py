import os
from box.exceptions import BoxValueError
import yaml
from sentimentanalyzer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import re, contractions, emoji
import pandas as pd
import json
import tensorflow as tf
import os


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


# src/sentimentanalyzer/utils/common.py

import csv
from pathlib import Path

def convert_to_csv(input_path: Path, output_path: Path):
    """
    Converts a FastText .ft.txt file into a CSV file with columns [label, text].

    Args:
        input_path (Path): Path to the .ft.txt input file.
        output_path (Path): Path to the CSV output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['label', 'text'])

        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            label = parts[0].replace('__label__', '')
            text = parts[1]
            writer.writerow([label, text])

def preprocess_review_list(df: pd.DataFrame):
    structured_data = []

    for _, row in df.iterrows():
        label = str(row['label']).replace('__label__', '')
        text = str(row['text'])

        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if sentence:
                structured_data.append({
                    'target': label,
                    'text': sentence.strip()
                })

    return pd.DataFrame(structured_data)


def clean_sentiment_text(text):
    # Import required libraries for text preprocessing
    import re, contractions, emoji

    # Convert all text to lowercase for normalization (e.g., "Happy" → "happy")
    text = text.lower()
    

    # Remove HTML tags (e.g., <div>, <br>, etc.)
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs (e.g., http://example.com or www.example.com)
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove email addresses (e.g., user@example.com)
    text = re.sub(r'\S+@\S+', '', text)



    # Remove unwanted special characters, keep letters, numbers, common punctuation, and emoji codes
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'":_]+', '', text)

    # Normalize whitespace: replace multiple spaces with a single space and trim ends
    text = re.sub(r'\s+', ' ', text).strip()

    # Return the cleaned and normalized text
    return text

    
import pandas as pd

def clean_target_column(df, target_col='target', text_col='text'):
    # Split the target column into numeric and text parts
    new_target = []
    new_text = []

    for val in df[target_col]:
        try:
            num, text = str(val).split(",", 1)
            new_target.append(int(num))
            new_text.append(text.strip().strip('"'))  # Remove leading/trailing quotes and whitespace
        except ValueError:
            # If splitting fails, assume bad format and handle gracefully
            new_target.append(None)
            new_text.append(str(val).strip().strip('"'))

    # Assign new cleaned columns
    df[target_col] = new_target
    df[text_col] = [f"{t} {orig}" if t else orig for t, orig in zip(new_text, df.get(text_col, ['']*len(df)))]

    return df


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")





import os
import tensorflow as tf

def get_element_spec(max_length):
    return (
        {
            'input_ids': tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

def load_datasets(transformer_data, max_length):
    element_spec = get_element_spec(max_length)

    train_tf = tf.data.experimental.load(os.path.join(transformer_data, "train_tf"), element_spec=element_spec)
    val_tf = tf.data.experimental.load(os.path.join(transformer_data, "val_tf"), element_spec=element_spec)
    test_tf = tf.data.experimental.load(os.path.join(transformer_data, "test_tf"), element_spec=element_spec)


    

    return train_tf, val_tf, test_tf

def set_seed(seed=42):
    tf.random.set_seed(seed)
