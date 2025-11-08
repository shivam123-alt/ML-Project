import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Splitting training and test input data.")
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor(),
            }
            
            # Hyperparameter grids for tuning each model
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [None, 10, 20]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "Linear Regression": {},  # no major hyperparameters
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "CatBoost": {
                    "iterations": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                }
            }
            best_models = {}
            best_scores = {}
            for name, model in models.items():
                param_grid = params.get(name, {})
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
                    gs.fit(X_train, y_train)
                    best_models[name] = gs.best_estimator_
                    best_scores[name] = gs.best_score_
                    logging.info(f"Best params for {name}: {gs.best_params_}")
                else:
                    model.fit(X_train, y_train)
                    score = model.score(X_train, y_train)
                    best_models[name] = model
                    best_scores[name] = score  # use training score as GridSearch not used
            
            # Evaluating model
            model_report:dict=evaluate_model(X_train=X_train,
                                             y_train=y_train,
                                             X_test=X_test,
                                             y_test=y_test,
                                             models=models
                                             )
            
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on the both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)
        