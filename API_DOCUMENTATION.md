# Loan Default Prediction API

 

## Notice
Due to billing constraints, this API is deployed on Render's free tier, which deactivates deployments after 15 minutes of inactivity. Please allow around 50 seconds for the initial request. Subsequent requests will be processed faster. We apologize for any inconvenience this may cause.

## Overview
This FastAPI application provides endpoints to predict loan defaults. The model and necessary transformations are applied to incoming data either as single data points or in bulk via **`CSV`** or **`Excel`** files.

## Enums
The following enumerations are used for categorical data:

### LocationEnum
- Beitbridge
- Harare
- Gweru
- Rusape
- Chipinge
- Chimanimani
- Marondera
- Kadoma
- Mutare
- Masvingo
- Bulawayo
- Kariba
- Plumtree
- Chiredzi
- Shurugwi
- Chivhu
- Zvishavane
- Nyanga
- Karoi
- Redcliff
- Kwekwe
- Gokwe
- Victoria Falls
- unknown
- Hwange

### JobEnum
- Teacher
- Nurse
- Doctor
- Data Analyst
- Software Developer
- Accountant
- Lawyer
- Engineer
- Data Scientist

### GenderEnum
- female
- other
- male

### MaritalStatusEnum
- married
- single
- divorced

## Endpoints

### `/predict`
- **Method:** POST
- **Description:** Predicts loan default for a single data point.

#### Request Body:
```json
{
  "gender": "GenderEnum",
  "is_employed": "bool",
  "job": "JobEnum",
  "location": "LocationEnum",
  "loan_amount": "float, >= 0",
  "number_of_defaults": "int, >= 0",
  "outstanding_balance": "float, >= 0",
  "interest_rate": "float, >= 0, <= 1",
  "age": "int, >= 0",
  "remaining_term": "int, >= 0",
  "salary": "float, >= 0",
  "marital_status": "MaritalStatusEnum"
}
```

## /predict-csv-excel

**Method: POST**
**Description**: Predicts loan defaults for data in a CSV or Excel file.

#### Request Body:
- file: A CSV or Excel file containing loan data.
#### Response:
- predictions: A list of predicted loan default statuses.


## Example Request:
 Upload a file with the following columns: gender, is_employed, job, location, loan_amount number_of_defaults, outstanding_balance, interest_rate, age, remaining_term, salary, marital_status.


Notes
Ensure the file contains all required columns and correct data types.
The model expects numerical and categorical features to be properly encoded. Transformation code should be added if needed.
Running the API
To start the FastAPI server, run the following command: