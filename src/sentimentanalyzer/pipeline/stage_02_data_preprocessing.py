from sentimentanalyzer.conponents.data_preprocessing import DataPreprocessing
from sentimentanalyzer.config.configuration import ConfigurationManager

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_preprocessing_config()
        data_preprocessor = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessor.preprocess()