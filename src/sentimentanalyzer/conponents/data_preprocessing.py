
import urllib.request as request
import zipfile
from sentimentanalyzer.logging import logger
from pathlib import Path
from sentimentanalyzer.entity import PreprocessingConfig
import pandas as pd
from pathlib import Path
from src.sentimentanalyzer.utils.common import convert_to_csv, preprocess_review_list, create_directories


class DataPreprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

        # Ensure output directory exists
        create_directories([Path(self.config.root_dir)])

    def preprocess(self):
        ingestion_dir = Path(self.config.ingestion_dir)
        output_dir    = Path(self.config.output_dir)

        # Find all .ft.txt files in ingestion_dir
        txt_files = list(ingestion_dir.glob("*.ft.txt"))
        if not txt_files:
            logger.warning(f"No .ft.txt files found in {ingestion_dir}")
            return

        for txt_path in txt_files:
            split_name = txt_path.stem.replace(".ft", "")  # e.g. 'train'
            csv_raw_path = output_dir / f"{split_name}_raw.csv"
            csv_cleaned  = output_dir / f"{split_name}_clean.csv"

            logger.info(f"Converting {txt_path.name} → {csv_raw_path.name}")
            convert_to_csv(txt_path, csv_raw_path)

            logger.info(f"Loading {csv_raw_path.name} into DataFrame")
            df_raw = pd.read_csv(csv_raw_path, low_memory=True)

            logger.info(f"Applying `preprocess_review_list` to {split_name}")
            df_clean = preprocess_review_list(df_raw)

            logger.info(f"Saving cleaned data to {csv_cleaned.name}")
            df_clean.to_csv(csv_cleaned, index=False)

        logger.info("Preprocessing complete.")
