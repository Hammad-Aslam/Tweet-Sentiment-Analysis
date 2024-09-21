
# Welcome to repository of Tweet Sentiment Analysis

This project aims to perform sentiment analysis on tweets using the Sentiment140 dataset. The objective is to classify tweets as either positive or negative based on their content, leveraging machine learning techniques for text classification.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Text Preprocessing](#text-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Serialization](#model-serialization)
- [Prediction](#prediction)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Sentiment analysis is a vital task in natural language processing (NLP) that helps in understanding the sentiment expressed in text data. This project focuses on analyzing tweets to determine their sentiment, which can be useful for various applications such as brand monitoring, market analysis, and public opinion tracking.

## Dataset

The Sentiment140 dataset is a collection of 1.6 million tweets labeled with sentiments. Each tweet is classified as positive (1) or negative (0). The dataset is available on Kaggle and provides a rich source of text data for sentiment analysis tasks.

## Installation

To run this project, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`
- `pickle`

You can install the required libraries using pip:

```bash
pip install numpy pandas nltk scikit-learn
```

Additionally, you will need a Kaggle account to access the dataset. Place your Kaggle API key (`kaggle.json`) in the project directory to enable data download.

## Project Structure

```
Twitter-Sentiment-Analysis/
│
├── kaggle.json          # Kaggle API key
├── sentiment140.zip     # Downloaded dataset
├── trained_model.sav    # Saved model after training
└── sentiment_analysis.ipynb  # Jupyter notebook for analysis
```

## Data Preparation

The first step involves downloading the dataset and extracting its contents. The following code accomplishes this:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kazanova/sentiment140
```

After downloading, the dataset is extracted, and the relevant CSV file is loaded into a pandas DataFrame for further processing.

## Text Preprocessing

Text preprocessing is crucial for improving the model's performance. In this step:

- Non-alphabetic characters are removed from the tweets.
- Tweets are converted to lowercase.
- Stopwords are eliminated, and words are stemmed to their root forms.

The following function performs these tasks:

```python
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
```

## Model Training

After preprocessing, the dataset is split into training and testing sets. TF-IDF vectorization is applied to convert the text data into a numerical format suitable for machine learning algorithms. A Logistic Regression model is then trained on the processed data:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

## Model Evaluation

The performance of the model is evaluated using accuracy scores on both the training and testing datasets. This provides insights into how well the model learned from the training data and how it generalizes to new, unseen data:

```python
from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score on the training data: ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Test data accuracy: ', test_data_accuracy)
```

## Model Serialization

To avoid retraining the model each time, it is saved using the pickle library. This allows for easy loading and reuse in future predictions:

```python
import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
```

## Prediction

With the trained model loaded, predictions can be made on new tweet data. The following code demonstrates how to make a prediction and interpret the results:

```python
X_new = X_test[200]
prediction = loaded_model.predict(X_new)

if prediction[0] == 0:
    print('Negative')
else:
    print('Positive')
```

## Future Work

Future enhancements for this project could include:

- Experimenting with different machine learning algorithms (e.g., SVM, Random Forest).
- Fine-tuning the hyperparameters of the Logistic Regression model.
- Implementing a user interface for real-time tweet sentiment analysis.
- Analyzing the impact of emojis and special characters on sentiment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This version maintains clarity and improves the overall flow. Let me know if you need further adjustments!
