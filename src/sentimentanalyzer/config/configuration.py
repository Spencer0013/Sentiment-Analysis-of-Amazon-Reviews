from sentimentanalyzer.constants import *
from dataclasses import dataclass
from sentimentanalyzer.utils.common import read_yaml, create_directories
from sentimentanalyzer.entity import (DataIngestionConfig,PreprocessingConfig, DataTransformationConfig,ModelTrainerConfig,ModelTrainerUSEConfig, EvaluationConfig)
import yaml
from sentimentanalyzer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from typing import Union
from pathlib import Path
                                   


# class ConfigurationManager:
#     def __init__(
#         self,
#         config_filepath = CONFIG_FILE_PATH,
#         params_filepath = PARAMS_FILE_PATH):

#         self.config = read_yaml(config_filepath)
#         self.params = read_yaml(params_filepath)

#         create_directories([self.config.artifacts_root])
    
class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Union[str, Path] = CONFIG_FILE_PATH,
        params_filepath: Union[str, Path] = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_preprocessing_config(self) -> PreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = PreprocessingConfig(
                      root_dir =config.root_dir,
                      ingestion_dir = config.ingestion_dir,
                      output_dir = config.output_dir
               )
        
        return data_preprocessing_config
    


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path_train=config.data_path_train,
            data_path_test = config.data_path_test,
            transformed_token_embedding_path=config.transformed_token_embedding_path,
            max_tokens=self.params.max_tokens,
            output_sequence_length=self.params.output_sequence_length,
            input_dim=self.params.input_dim,
            output_dim=self.params.output_dim,
            batch_size=self.params.batch_size
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Read the `model_trainer` section of the config and
        combine it with training params into a ModelTrainerConfig.
        """
        trainer_cfg = self.config.model_trainer

        # make sure the model‐trainer folder exists
        create_directories([trainer_cfg.root_dir])

        return ModelTrainerConfig(
            root_dir=trainer_cfg.root_dir,
            model_save_path=trainer_cfg.model_save_path,
            epochs=self.params.epochs,
            classes=self.params.classes,
            learning_rate=self.params.learning_rate,
            input_shape=self.params.input_shape,
            input_dtype=self.params.input_dtype,
            params=self.params       # if your dataclass includes this field
        )
    
    def get_model_trainer_use_config(self) -> ModelTrainerUSEConfig:
        config = self.config.model_trainer_use
        if config is None:
            raise ValueError("Missing 'model_trainer_use' section in config file.")

        create_directories([config.root_dir])

        return ModelTrainerUSEConfig(
            root_dir=config.root_dir,
            use_model_path =config.use_model_path,
            data_path = config.data_path,
            classes=self.params.classes,
            model_save_path=config.model_save_path,
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            learning_rate=self.params.learning_rate
        )

   
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/model_trainer_USE/model.h5",
            test_data="artifacts/model_trainer_USE/test.tfrecord",
            all_params=self.params
        )
        return eval_config
