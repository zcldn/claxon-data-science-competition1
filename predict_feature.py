from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import xgboost as xgb
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.linear_model import LassoCV

#fitting random forest to my model 
clf = RandomForestClassifier()
clf = xgb.XGBClassifier()
# clf = CatBoostClassifier()

model =     RandomForestClassifier(n_jobs=-1)

class Predict_Marital:
   

    def __init__(self, data, model=clf):
        self.predict_values = data.loc[data['marital_status'] == ' '].drop(columns = ['marital_status'])
        self.features = data.loc[data['marital_status'] != ' '].drop(columns = 'marital_status')
        self.target = data.loc[data['marital_status'] != ' ','marital_status']
        self.model = model

    def preprocess(self):
    
        
        self.target = self.target.map({'single':0, 'married':1, 'divorced': 2}).astype('int')
        
        
        
        return self.features, self.target
    

    def train(self):
        features, target = self.preprocess()
        X_train, X_test, y_train,y_test = train_test_split(features,target, test_size=0.2)
        
        
        self.model.fit(X_train, y_train)
        print(f'score: {self.model.score(X_test, y_test)}')

        
        return self
    
    def predict(self):

        
        preds = self.model.predict(self.predict_values)
        feature_importances = pd.DataFrame(self.model.feature_importances_).T
        feature_importances.columns = self.model.feature_names_in_
        feature_importances = feature_importances.T.sort_values(by=0)

        return preds, feature_importances


    
    
 


