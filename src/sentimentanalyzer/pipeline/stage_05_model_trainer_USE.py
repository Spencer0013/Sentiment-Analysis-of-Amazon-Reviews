from sentimentanalyzer.config.configuration import ConfigurationManager
from sentimentanalyzer.conponents.model_trainer_use import ModelTrainerUSE

class ModelTrainerUSEPipeline:
    def __init__(self):
        pass

    def main(self):
          config = ConfigurationManager()
          model_trainer_use_config = config.get_model_trainer_use_config()
          model_trainer_use = ModelTrainerUSE(config=model_trainer_use_config)
          model_trainer_use.build_model()
          model_trainer_use.load_data()
          model_trainer_use.load_and_encode_data()
          model_trainer_use.prepare_datasets()
          model_trainer_use.train()
          model_trainer_use.save_tf_datasets()