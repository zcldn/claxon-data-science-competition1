import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

# Custom Classes
class Predict_Marital(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        if model is None:
            model = CatBoostClassifier( verbose=0)
        self.model = model

    def fit(self, X, y=None):
        features = X.loc[X['marital_status'] != ' '].drop(columns='marital_status')
        target = X.loc[X['marital_status'] != ' ', 'marital_status']
        target = target.map({'single': 0, 'married': 1, 'divorced': 2}).astype('int')
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=84, stratify=target)
        self.model.fit(X_train, y_train)
        print(f'Predict_Marital Model Score: {self.model.score(X_test, y_test)}')
        return self

    def transform(self, X):
        predict_values = X.loc[X['marital_status'] == ' '].drop(columns=['marital_status'])
        preds = self.model.predict(predict_values)
        X.loc[X['marital_status'] == ' ', 'marital_status'] = pd.DataFrame(preds)[0].map({0: 'single', 1: 'married', 2: 'divorced'}).to_numpy()
        X['marital_status'] = X['marital_status'].map({'single': 0, 'married': 1, 'divorced': 2}).astype('int')
        return X

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, labels, threshold=0.2):
        self.label_encoders = {}
        self.features = features
        self.labels = labels
        self.location_loan_index_encode = [{13: 0, 12: 1, 3: 2, 4: 3, 18: 4, 21: 5, 20: 6, 6: 7, 15: 8, 24: 9, 16: 10, 22: 11, 8: 12, 1: 13, 9: 14, 7: 15, 14: 16, 5: 17, 0: 18, 2: 19, 10: 20, 19: 21, 17: 22, 23: 23, 11: 24}]
        self.job_loan_index_encode = [{4: 0, 7: 1, 3: 2, 6: 3, 0: 4, 1: 5, 5: 6, 8: 7, 2: 8}]
        self.location_ratios = None
        self.threshold = threshold
        self.isFitted = False
        self.label_encodings = {}
        self.preprocessor = None

    def fit(self, X, y=None):

        numerical_cols = ['remaining term', 'outstanding_balance', 'interest_rate', 'age', 'loan_amount', 'salary', 'is_employed']
        categorical_cols = ['gender', 'marital_status', 'number_of_defaults', 'job', 'location']

        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),


        ])



        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        # Bundle preprocessing for numerical and categorical data

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        self.preprocessor = preprocessor.fit(X)
        print(self.preprocessor)
        categorical_features = ['job', 'location']
        for cat in categorical_features:
            le = LabelEncoder()
            
            le.fit(X[cat])
            self.label_encoders[cat] = le
            self.label_encodings[cat] = dict(zip(le.classes_, le.transform(le.classes_)))

        X_features = self.transform(self.features.copy())[['location']]
        X_features['y'] = self.labels.map({'Did not default': 0, 'Defaulted': 1})
        self.location_ratios = X_features.groupby('location')['y'].mean()
        self.location_ratios = self.location_ratios / (1 - self.location_ratios)
        return self

    def categorize_location(self,location):

        if location in ['Karoi', 'Hwange', 'Chiredzi', 'Gokwe', 'Shurugwi', 'Victoria Falls']:
            return 2
        elif location in ['Nyanga', 'Chivhu', 'Kadoma', 'Marondera', 'Gweru', 'Masvingo', 'Harare', 'Bulawayo', 'Mutare']:
            return 0
        else:
            return 1

        # Apply the function to the 'location' column

    

    def transform(self, X):
    # Apply preprocessing steps defined in `fit` method
        X = self.preprocessor.transform(X)

        # Convert to DataFrame with appropriate columns
        features = ['remaining term', 'outstanding_balance', 'interest_rate', 'age', 'loan_amount', 'salary', 'is_employed',
                    'gender', 'marital_status', 'number_of_defaults', 'job', 'location']
        X = pd.DataFrame(X, columns=features)

        # Apply custom transformations
        X = X.assign(Category=X['location'].apply(self.categorize_location))
        
        # Encode categorical features
        categorical_features = ['job', 'location']
        for cat in categorical_features:
            X[cat] = self.label_encoders[cat].transform(X[cat])
        
        X['gender'] = X['gender'].map({"other": 1, "female": 0, 'male': 2}).astype('int')
        X['location'] = X['location'].map(self.location_loan_index_encode[0])
        X['job'] = X['job'].map(self.job_loan_index_encode[0]).astype('int')

        # Independent binary features for loan_amount
        X['loan_equal_5000'] = np.where(X.loan_amount == 5000, 1, 0)
        X['loan_>_75000'] = np.where(X.loan_amount > 75000, 1, 0)
        X = X.assign(total_loan_amount_per_job=X.groupby('job')['loan_amount'].transform('mean'))

        # Create new feature to capture job and location relationship
        X['job_location_interact'] = np.log1p(np.sqrt(X['job'])) / (X['location'] + 1)

        if self.isFitted:
            X['location_class'] = X['location'].map(lambda loc: 1 if self.location_ratios[loc] < self.threshold else 0)

        X = self.create_new_features(X)
        return X


      

    def create_new_features(self, X):
        epsilon = 1e-8  # Small value to avoid division by zero
        X = X.copy()  # To avoid modifying the original DataFrame
        X['interest_rate_per_loan_amount'] = X['interest_rate'] / (X['total_loan_amount_per_job'] + epsilon)
        X['age_per_loan_amount'] = X['age'] / (X['total_loan_amount_per_job'] + epsilon)
        X['job_location_per_loan_amount'] = X['job_location_interact'] / (X['total_loan_amount_per_job'] + epsilon)
        X['loan_5000_and_loan_75000_interaction'] = X['loan_equal_5000'] * X['loan_>_75000']
        return X

# Data Preparation
