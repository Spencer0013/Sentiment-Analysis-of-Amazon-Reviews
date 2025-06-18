#from sentimentanalyzer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
#from sentimentanalyzer.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
#from sentimentanalyzer.pipeline.stage_03_data_transformation import DataTransformationPipeline
#from sentimentanalyzer.pipeline.stage_04_model_trainer import ModelTrainerPipeline
#from sentimentanalyzer.pipeline.stage_05_model_trainer_USE import ModelTrainerUSEPipeline
from sentimentanalyzer.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from src.sentimentanalyzer.logging import logger

# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Data Preprocessing stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_processor = DataPreprocessingPipeline()
#    data_processor.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Data Transformation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_transformation = DataTransformationPipeline()
#    data_transformation.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Model Trainer stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    model_trainer = ModelTrainerPipeline()
#    model_trainer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STAGE_NAME = "Model Trainer USE stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    model_trainer_use = ModelTrainerUSEPipeline()
#    model_trainer_use.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e



STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   evaluation = ModelEvaluationPipeline()
   evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




