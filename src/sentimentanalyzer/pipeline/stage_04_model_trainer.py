from sentimentanalyzer.config.configuration import ConfigurationManager
from sentimentanalyzer.conponents.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer = model_trainer.train()