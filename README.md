## Disaster Response Pipeline

![image info](./images/header.jfif)

In this project using two datasets of distress mesages from three different channels, their response categories (36) and data, NLP and Machine learning pipelines we build an web application with the optimized classifier to predict in which response category the message is likelier to belong, reducing the potential reaction time of the responding organizations.

## Installation

This project requires Python 3.x and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Json
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sys
- Re
- Pickle


## Files Description

        disaster-response
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- messagesDB.db
                |-- process_data.py
          |-- images
                |-- header.jfif
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- notebooks
                |-- ETL Pipeline Preparation.ipynb
                |-- ML Pipeline Preparation.ipynb
          |-- README
          |-- LICENSE


## ETL Pipeline
File data/process_data.py contains data cleaning pipeline that:

- Loads the messages and categories dataset
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

## ML Pipeline
File models/train_classifier.py contains machine learning pipeline that:

- Loads data from the SQLite database
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final and best model as a pickle file

## Flask Web App

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/messagesDB.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Web Application Screenshots

Visual 1

![image info](./images/graph1.jpg)

Visual 2

![image info](./images/graph2.jpg)

Visual 3

![image info](./images/graph3.jpg)


## License

Apache License</p>
Version 2.0, January 2004</p>
http://www.apache.org/licenses/


## Conclusions
Since the dataset is unbalanced (Visual 2), we will proceed to use the 'macro average' for the model evaluation metrics. Hence the macro-average gives every class the same importance, and therefore better reflects how well the model performs.
