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
    data_path_train: str
    data_path_test: str
    transformed_token_embedding_path: str
    max_tokens: int
    output_sequence_length: int
    input_dim: int
    output_dim: int
    batch_size: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_save_path: Path
    epochs:int
    classes:int
    learning_rate:float
    input_shape:tuple
    input_dtype: int
    params: any

@dataclass
class ModelTrainerUSEConfig:
    root_dir: Path
    use_model_path: str
    data_path: Path
    classes: int
    model_save_path: Path
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    test_data: Path
    all_params: dict

