import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean
import sqlalchemy
import re
import nltk
import time
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='messages', con = engine)

    #split dataset in features(X) and target (Y)
    X = df['message']
    Y = df.iloc[:,4:]
    
    category_names = Y.columns
    
    #Downloading nltk packages
    nltk.download(['punkt', 'wordnet','stopwords'])
    
    return X,Y, category_names
    
def tokenize(text):
    """
    Function for converting messages text in tokens
    input:
        text message
    output:
        Remove stop words
        word tokenize, lemmatize
        
    """    
    stop_words = set(stopwords.words('english'))     
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # normalize case, remove punctuation and droping numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = ''.join((x for x in text if not x.isdigit()))
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    #Using AdaBoostClassifer
    pipeline_Ada = Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer()),
                ('clf',MultiOutputClassifier(AdaBoostClassifier())),
    ])

    parameters = {'vect__max_df': [0.75],
                  'tfidf__use_idf': [False]}

    cv_ada = GridSearchCV(pipeline_Ada,param_grid=parameters, cv = 3)
    
    return cv_ada


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function iterates through each category and collect precision, recall and Fscore metric into 
    a DataFrame
    input:
        model: AdaBoosClassifier
    
    output:
        print average metric from all categories
    """
    
    y_pred_ada = model.predict(X_test)
    
    metrics = []
    for idx, col in enumerate(category_names):
        precision,recall,fscore,support=score(Y_test.iloc[:,idx], y_pred_ada[:,idx], average='macro')
        metrics.append([col,precision,recall,fscore])
    
    df_metrics =  pd.DataFrame(metrics,columns=['Category','Precision','Recall','Fscore'])
    
    print('Total evaluation over all categories:\n',
      ' Precision: {:.4f}, Recall: {:.4f}, FScore: {:.4f}'.format(df_metrics['Precision'].mean(),
                                                     df_metrics['Recall'].mean(),
                                                     df_metrics['Fscore'].mean()))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()