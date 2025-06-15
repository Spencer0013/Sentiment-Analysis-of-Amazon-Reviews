from sentimentanalyzer.config.configuration import ConfigurationManager
from sentimentanalyzer.conponents.model_evaluation import Evaluation

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_evaluation_config()
        evaluation = Evaluation(val_config)
        evaluation.load_model()
        evaluation.load_test_dataset()
        evaluation.evaluate()
        evaluation.save_score()


