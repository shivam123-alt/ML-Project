import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
        
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
            
     

class CustomData:
    def __init__(self,
                 gender:str,
                 reace_ethnicity: str,
                 parental_level_of_education,
                 lunch:str,
                 test_prepration_course:str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        
        self.reace_ethnicity = reace_ethnicity
        
        self.parental_level_of_education=parental_level_of_education
        
        self.lunch=lunch
        
        self.test_prepration_course=test_prepration_course
        
        self.reading_score=reading_score
        
        self.writing_score=writing_score
        
    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race_ethnicity":[self.reace_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_prepration_course]":[self.test_prepration_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
                
            }
            return pd.DataFrame(custom_data_input_dict)
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
