import re 
import nltk
import spacy
import pickle
import uvicorn
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords   
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')


class model:
    def read_dataset(data):
        dataset = pd.read_csv(data)
        dataset.drop('Unnamed: 0', axis = 1, inplace = True)
        dataset.columns = ['sentiment', 'review']
        dataset.review = dataset.review.str.split(" ",1).str[1]     #removing first word of each review (@airline_name)
        encoded_dict = {'negative': 0, 'positive': 1}               
        dataset['sentiment'] = dataset.sentiment.map(encoded_dict)
        return dataset


    def clean_dataset(column):
        stop_words = stopwords.words('english')
        stop_words.remove('not')
        stop_words.remove('or')
        for row in column:
        # Split CamelCase words
            row = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', str(row))).split()
        # Remove special characters and numbers 
            row = re.sub('[^a-zA-Z]', ' ', str(row))
        # Remove Repeated words
            row = re.sub(r"\\b(\\w+)(?:\\W+\\1\\b)+", "", str(row))
        # Replace tabs and newlines with a single space
            row = re.sub("(\\t)", " ", str(row))
            row = re.sub("(\\r)", " ", str(row))
            row = re.sub("(\\n)", " ", str(row))
        # Remove single alphabets
            row = re.sub(r'(?:^| )\w(?:$| )', ' ', str(row)).strip()
            row = row.split()
            row = [word for word in row if not word in set(stop_words)]
            row = ' '.join(row)
            row = row.lower()
            yield row

    
    def reviews(data):
        dataset = model.read_dataset(data)
        reviews = model.clean_dataset(dataset.review)
        nlp = spacy.load("en_core_web_sm", disable = ['ner', 'parser'])
        reviews = [str(doc) for doc in nlp.pipe(reviews, batch_size = 128)]
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(reviews).toarray()
        y = dataset.sentiment
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        return cv, X_train, X_test, y_train, y_test

    def train(X_train, y_train):
        classifier = LogisticRegression(random_state = 0, C = 1.0, penalty = 'l2', solver = 'saga')
        classifier.fit(X_train, y_train)
        return classifier

data = 'airline_sentiment_analysis.csv'
cv, X_train, X_test, y_train, y_test = model.reviews(data)
classifier = model.train(X_train, y_train)

y_pred = classifier.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")

app = FastAPI()

class Dataset(BaseModel):
    review: str

@app.get('/')
def index():
    return {'message': 'Go to localhost:8000/docs to see swagger documentation'}

@app.post('/predict')
def predict_sentiment(data: Dataset):
    data = data.dict()
    newreview = data['review']
    newreview = model.clean_dataset([newreview])
    pred = classifier.predict(cv.transform(newreview).toarray())
    if pred == 1:
        pred = 'Positive'
    else:
        pred = 'Negative'
    return {'Model Prediction': pred}

if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = 8000)