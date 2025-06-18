from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class PreprocessingConfig:
    root_dir: Path
    ingestion_dir: Path
    output_dir: Path




@dataclass
class DataTransformationConfig:
    root_dir: str
    data_path: str
    

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_save_path: Path
    epochs:int
    classes:int
    learning_rate:float
    input_dtype: int
    params: any
    random_state:int
    max_tokens: int
    output_sequence_length: int
    input_dim: int
    output_dim: int
    batch_size: int
    label_col: str




@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    data_path: Path
    all_params: dict

