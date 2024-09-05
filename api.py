from enum import Enum
import joblib
import numpy as np
from pydantic import BaseModel, conint, confloat
import pandas as pd
from joblib import load
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union

import uvicorn

# Define enums for location, job, gender, and marital status
class LocationEnum(str, Enum):
    Beitbridge = 'Beitbridge'
    Harare = 'Harare'
    Gweru = 'Gweru'
    Rusape = 'Rusape'
    Chipinge = 'Chipinge'
    Chimanimani = 'Chimanimani'
    Marondera = 'Marondera'
    Kadoma = 'Kadoma'
    Mutare = 'Mutare'
    Masvingo = 'Masvingo'
    Bulawayo = 'Bulawayo'
    Kariba = 'Kariba'
    Plumtree = 'Plumtree'
    Chiredzi = 'Chiredzi'
    Shurugwi = 'Shurugwi'
    Chivhu = 'Chivhu'
    Zvishavane = 'Zvishavane'
    Nyanga = 'Nyanga'
    Karoi = 'Karoi'
    Redcliff = 'Redcliff'
    Kwekwe = 'Kwekwe'
    Gokwe = 'Gokwe'
    Victoria_Falls = 'Victoria Falls'
    unknown = 'unknown'
    Hwange = 'Hwange'

class JobEnum(str, Enum):
    Teacher = 'Teacher'
    Nurse = 'Nurse'
    Doctor = 'Doctor'
    Data_Analyst = 'Data Analyst'
    Software_Developer = 'Software Developer'
    Accountant = 'Accountant'
    Lawyer = 'Lawyer'
    Engineer = 'Engineer'
    Data_Scientist = 'Data Scientist'

class GenderEnum(str, Enum):
    female = 'female'
    other = 'other'
    male = 'male'

class MaritalStatusEnum(str, Enum):
    married = 'married'
    single = 'single'
    divorced = 'divorced'

class LoanData(BaseModel):
    gender: Optional[GenderEnum]
    is_employed: Optional[bool]
    job: Optional[JobEnum]
    location: Optional[LocationEnum]
    loan_amount: Optional[confloat(ge=0)]
    number_of_defaults: Optional[conint(ge=0)]
    outstanding_balance: Optional[confloat(ge=0)]
    interest_rate: Optional[confloat(ge=0, le=1)]
    age: Optional[conint(ge=0)]
    remaining_term: Optional[conint(ge=0)]
    salary: Optional[confloat(ge=0)]
    marital_status: Optional[MaritalStatusEnum]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and transformer using joblib
model = joblib.load('model_file.pkl')

@app.post('/predict')
async def predict(data: LoanData):
    """
    Predict loan default for a single data point.
    """
    try:
        # Process single data point input
        point = {
            "gender": data.gender.value,
            "is_employed": data.is_employed,
            "job": data.job.value,
            "location": data.location.value,
            "loan_amount": data.loan_amount,
            "number_of_defaults": data.number_of_defaults,
            "outstanding_balance": data.outstanding_balance,
            "interest_rate": data.interest_rate,
            "age": data.age,
            "remaining term": data.remaining_term,
            "salary": data.salary,
            "marital_status": data.marital_status.value
        }

        # Convert the input data to a DataFrame
        df = pd.DataFrame([point])

        # Make prediction
        pred = model.predict(df)
        pred = np.where(pred == 1, "Defaulted", "Did not default")
        return {"prediction": pred.tolist()[0]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/predict-csv-excel')
async def predict_csv_excel(file: UploadFile = File(...)):
    """
    Predict loan defaults for data in a CSV or Excel file.
    """
    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only CSV and Excel files are accepted.")

        # Ensure the DataFrame has the correct columns
        required_columns = [
            "gender", "is_employed", "job", "location", "loan_amount",
            "number_of_defaults", "outstanding_balance", "interest_rate",
            "age", "remaining term", "salary", "marital_status"  # Updated column name
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")
        
        df = df[required_columns]

        # Ensure proper types for numeric columns
        df = df.astype({
            "is_employed": bool,
            "loan_amount": float,
            "number_of_defaults": int,
            "outstanding_balance": float,
            "interest_rate": float,
            "age": int,
            "remaining term": int,  # Updated column name
            "salary": float
        })

        # Transform the data if needed (e.g., encoding categorical variables)
        # Transformations should be added here if required

        # Make predictions
        preds = np.where(model.predict(df)==1, 'Defaulted', 'Did not default')


        # Return predictions as a list
        return {"predictions": preds.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

   