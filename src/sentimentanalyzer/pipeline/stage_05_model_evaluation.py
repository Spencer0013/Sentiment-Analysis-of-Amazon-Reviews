from sentimentanalyzer.config.configuration import ConfigurationManager
from sentimentanalyzer.conponents.model_evaluation import Evaluation

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.load_model()
        evaluation.evaluate()
        evaluation.save_score()

        


