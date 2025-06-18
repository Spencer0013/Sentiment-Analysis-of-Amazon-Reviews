
import pandas as pd
import numpy as np
import re, contractions, emoji
from sentimentanalyzer.logging import logger
from pathlib import Path
from sentimentanalyzer.entity import PreprocessingConfig
from src.sentimentanalyzer.utils.common import  create_directories,  get_labels_and_texts_from_txt


class DataPreprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

        # Ensure output directory exists
        create_directories([Path(self.config.root_dir)])

        ingestion_dir = Path(self.config.ingestion_dir)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = {
            "train": ingestion_dir / "train.ft.txt",
            "test": ingestion_dir / "test.ft.txt"
        }

        for split, filepath in datasets.items():
            if filepath.exists():
                labels, texts = get_labels_and_texts_from_txt(filepath)

                # Save with split in filename
                labels_path = output_dir / f"{split}_labels.npy"
                texts_path = output_dir / f"{split}_texts.txt"

                np.save(labels_path, labels)
                with open(texts_path, "w", encoding="utf-8") as f_out:
                    for text in texts:
                        f_out.write(text + "\n")

                print(f"{split.capitalize()} set saved:")
                print(f"✅ Labels -> {labels_path}")
                print(f"✅ Texts  -> {texts_path}")
            else:
                print(f"⚠️ No file found for {split} at {filepath}")
