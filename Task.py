#!/usr/bin/env python
# coding: utf-8

# # DataCoLab Task Assignment

## Step 1: Data Preprocessing and EDA

# import google drive to use gpu for this section
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import re
import json
import string
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from nltk.corpus import stopwords

# CONSTANTS
UNKNOWN_GENRE = ('Unknown',)
UNKNOWN_AUTHOR = 'Unknown'

"""### Load dataset"""

# header for dataset
header = ['Wikipedia_Article_ID', 'Freebase_ID', 'Book_Title', 'Author',
          'Publication_date', 'Book_Genres_(Freebase_ID:name_tuples)',
          'Plot_Summary']

# Load dataset from text file in google drive
file_path = 'drive/MyDrive/booksummaries.txt'
data = pd.read_csv(file_path, sep='\t', header=None,
                   names=header, encoding='utf-8')

# # header for dataset
# header = ['Wikipedia_Article_ID', 'Freebase_ID', 'Book_Title', 'Author',
#           'Publication_date', 'Book_Genres_(Freebase_ID:name_tuples)',
#           'Plot_Summary']

# # Load dataset from text file in local system
# data = pd.read_csv('booksummaries.txt', sep='\t', header=None,
#                    names=header, encoding='utf-8')

# Display sample of data
data.head()

# Display shape of data
print(f"Shape of dataset: {data.shape}")

"""### Preprocess Section"""

# # Drop "Wikipedia article ID", "Freebase_ID" column cause that values are
# # unique and useless
data = data.drop("Wikipedia_Article_ID", axis=1)
data = data.drop("Freebase_ID", axis=1)

# Add "Index" column by reset it
data = data.reset_index()
data.rename(columns={'index': 'Index'}, inplace=True)

# convert all genres values into tuple
for i in tqdm(range(len(data))):
    try:
        genres_dict = json.loads(
            data.loc[i, 'Book_Genres_(Freebase_ID:name_tuples)'])
        if isinstance(genres_dict, dict):
            data.at[i, 'Book_Genres_(Freebase_ID:name_tuples)'] = tuple(
                genres_dict.values())
        else:
            data.at[i, 'Book_Genres_(Freebase_ID:name_tuples)'] = np.nan
    # Handle NaN values and JSON decoding errors
    except (TypeError, json.JSONDecodeError):
        continue

data.rename(columns={'Book_Genres_(Freebase_ID:name_tuples)': 'Book_Genres'},
            inplace=True)

"""### Predict book genres uses ML"""

# import libraries
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Preprocess plot summaries
def preprocess_text(text):
    """
    Preprocess plot summaries
    """
    # # Flatten tuples if necessary
    # if isinstance(text, tuple):
    #     text = ' '.join(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])

    # Stem words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text

for index, row in tqdm(data.iterrows()):
    data.loc[index, 'Clean_Plot_Summary'] = preprocess_text(row['Plot_Summary'])

# save checkpoint_book_summary
file_path = 'drive/MyDrive/checkpoint_book_summary.csv'
data.to_csv(file_path, sep='\t',
            index=False, encoding='utf-8')

# Load dataset from csv file
file_path = 'drive/MyDrive/checkpoint_book_summary.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

data[5:15]

# from here because we want to use ML, we get copy from data
df = data.copy(deep=True)

# Filter rows with non-null genres
df = df[df['Book_Genres'].notnull()]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Clean_Plot_Summary'])

# Extract genre labels from tuples (if applicable)
if isinstance(df['Book_Genres'][0], tuple):
    df['Book_Genres'] = df['Book_Genres'].apply(lambda x: '|'.join(x))

# Create a label encoder to handle multi-word genres
le = LabelEncoder()
y_encoded = le.fit_transform(df['Book_Genres'])

# Create a one-hot encoder
ohe = OneHotEncoder(sparse_output=False)
y_onehot = ohe.fit_transform(y_encoded.reshape(-1, 1))

# Split into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print(f"X_train shape:         {X_train.shape}")
print(f"y_train_encoded shape: {y_train_encoded.shape}")
print(f"X_test shape:          {X_test.shape}")
print(f"y_test_encoded shape:  {y_test_encoded.shape}")

# Extract class labels from one-hot encoding
# (assuming each row represents a single genre)
# Get the index of the maximum value in each row
y_train = y_train_encoded.argmax(axis=1)
y_test = y_test_encoded.argmax(axis=1)

# Create and train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict genres for the testing set
y_pred_encoded = clf.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict genres for books with missing values
missing_genres_index = data[data['Book_Genres'].isnull()].index
X_missing = vectorizer.transform(data.loc[missing_genres_index, 'Clean_Plot_Summary'])
predicted_genres = clf.predict(X_missing)

# Fill in missing genres
df.loc[missing_genres_index, 'Book genres'] = le.inverse_transform(predicted_genres)

"""#### method 2 for predicted genres"""

# get the distinct book genres
unique_genres = dict()
for index, row in data.iterrows():
    if pd.isnull(row['Book_Genres']):
        continue
    for genre in row['Book_Genres']:
        unique_genres[genre] = unique_genres.get(genre, 0) + 1

top_genres = [k for k, v in sorted(unique_genres.items(), key=lambda x: x[1], reverse=True) if v > 50]

total_genres = [k for k, v in sorted(unique_genres.items(), key=lambda x: x[1], reverse=True)]

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_genres))

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=len(total_genres))

data.head(15)

data.loc[7, 'Cleaned Summary']

# Preprocess plot summary
text = data.loc[7, 'Plot summary']
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# Map predicted index to genre
predicted_genre = unique_genres[predictions.item()]
print("Predicted genre:", predicted_genre)

"""#### method 3: train bert model based on my data"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

# Load dataset from csv file from drive for google colab
file_path = 'drive/MyDrive/summarized_book_summary.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

# Preprocessing: Genres are stored as tuples
# Convert string representation of tuples into actual tuples
data['Book genres'] = data['Book genres'].apply(lambda x: eval(x) if pd.notnull(x) else x)

for index, row in data.iterrows():
    if row['Book genres'][0] == 'Unknown':
        data.loc[index, 'Book genres'] = np.nan

df = data.copy(deep=True)

# Filter rows where genres are not null for training
df_train = df[df['Book genres'].notnull()]

# Binarize the genres using MultiLabelBinarizer for training data
mlb = MultiLabelBinarizer()
genres_binarized = mlb.fit_transform(df_train['Book genres'])  # Transform tuples into binary vectors

# Prepare Dataset Class
class BookDataset(Dataset):
    def __init__(self, summaries, labels, tokenizer, max_len):
        self.summaries = summaries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = str(self.summaries[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'summary': summary,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Set up Dataset and DataLoader
MAX_LEN = 512
BATCH_SIZE = 16

dataset = BookDataset(
    summaries=df_train['Summarized_Text'].values,
    labels=genres_binarized,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define optimizer, loss function (Binary Cross Entropy for multi-label), and scheduler
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

# Training Loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)

# Train the model (you can further enhance with validation logic)
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss = train_epoch(model, dataloader, loss_fn, optimizer, device)
    print(f'Train loss: {train_loss}')

# Save the model after training
model_path = "drive/MyDrive/bert_trained_model"
model.save_pretrained(model_path)

# Save the tokenizer as well
tokenizer.save_pretrained(model_path)

# Predict missing genres (incomplete entries)
def predict_genres(model, summary, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        summary,
        max_length=max_len,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    probabilities = torch.sigmoid(output.logits)

    return probabilities.detach().cpu().numpy()

# Predict missing genres and fill missing values
for idx, row in data[data['Book genres'].isna()].iterrows():
    summary = row['Summarized_Text']
    predicted_probs = predict_genres(model, summary, tokenizer, MAX_LEN)
    predicted_labels = mlb.inverse_transform(predicted_probs > 0.1)  # Set a threshold
    df.at[idx, 'Book genres'] = predicted_labels[0]  # Assign the predicted tuple of genres

# Save the updated dataset
file_path = 'drive/MyDrive/genre_filled_dataset.csv'
data.to_csv(file_path, sep='\t',
            index=False, encoding='utf-8')

data.head(20)

"""### Fill Author and Publication date missing values with Google Book API"""

import os
import requests
import numpy as np

def get_book_info(title, author=None):
    """Fetches book information from the Google Books API for a given title and author.

    Args:
        title (str): The title of the book to search for.
        author (str, optional): The author of the book. If provided, the search query will be refined.

    Returns:
        dict: A dictionary containing book information (title, author, publication date)
            if found. Returns None if no results are found.

    Raises:
        Exception: Raises an exception if there's an error fetching data from the API.
    """

    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if not api_key:
      raise ValueError("Missing environment variable: GOOGLE_BOOKS_API_KEY")

    query_params = {"q": title}
    if author:
      query_params["author"] = author

    url = f"https://www.googleapis.com/books/v1/volumes?{urlencode(query_params)}&maxResults=1&key={api_key}"

    try:
      response = requests.get(url)
      response.raise_for_status()  # Raise an exception for non-200 status codes
    except requests.exceptions.RequestException as e:
      raise Exception(f"Error fetching book information: {e}") from e

    data = response.json()
    if 'items' in data:
      book_data = data['items'][0]['volumeInfo']
      return {
        'title': book_data['title'],
        'author': book_data['authors'][0] if 'authors' in book_data else np.nan,
        'publication_date': book_data['publishedDate'] if 'publishedDate' in book_data else np.nan,
      }
    else:
      return None

# Fill Author and Publication date missing values with Google Book API
for index, row in data.iterrows():
    # Check if either Author or Publication date is missing
    if pd.isnull(row['Author']) or pd.isnull(row['Publication date']):
        # Use the existing author if available, otherwise send None
        author = row['Author'] if not pd.isnull(row['Author']) else None
        book_info = get_book_info(row['Book title'], author)
        if book_info:
            data.at[index, 'Author'] = book_info['author']
            data.at[index, 'Publication date'] = book_info['publication_date']

"""### end of handle missing values"""

def extract_year(pub_date):
    """
    Convert publication date to just the year
    (if it's in YYYY-MM-DD format)
    """
    if isinstance(pub_date, str) and '-' in pub_date:
        return pd.to_datetime(pub_date, errors='coerce').year
    return int(pub_date) if isinstance(pub_date, str) and pub_date.isdigit()\
        else np.nan

data['Publication year'] = data['Publication date'].apply(extract_year)

# Remove duplicate rows
# Check for duplicates based on a unique identifier or key columns
duplicates = data.duplicated(subset=['Book title', 'Author'], keep='first')
print(f"Number of duplicate rows: {duplicates.sum()}")

# # Drop duplicates
data = data.drop_duplicates(subset=['Book title', 'Author'], keep='first')

# text cleaning
# if get an error for this line, you must use `nltk.download()`
# command and download stopwords
# stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stop words
    # we don't remove stop words to get more accurate text in step 2
    # text = ' '.join([word for word in text.split()
    #                  if word not in stop_words])
    # Remove extra  before and after whitespace

    text = text.strip()
    # check if text is none
    text = text if text else np.nan

    return text


data['Cleaned_Summary'] = data['Plot_Summary'].apply(clean_text)

# Drop rows where 'Cleaned Summary' is NaN
data = data.dropna(subset=['Cleaned Summary'])

# summary length
data['Summary Length'] = data['Cleaned Summary']\
    .apply(lambda x: len(x.split()))

# Detect outliers in Summary_Length
Q1 = data['Summary Length'].quantile(0.25)
Q3 = data['Summary Length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['Summary Length'] < lower_bound) |
                (data['Summary Length'] > upper_bound)]
print(f"Number of outliers in Summary Length: {outliers.shape[0]}")

# Remove outliers
data = data[(data['Summary Length'] >= lower_bound) &
            (data['Summary Length'] <= upper_bound)]

# Fill Author missing values with UNKNOWN_AUTHOR variable because
# We cannot put anyone's book under the name of another author
data['Author'].fillna(UNKNOWN_AUTHOR, inplace=True)

# Fill missing values book generes based of other author's book
# this section of code is not good and fix just 2 rows:)))


def fill_missing_genres(df, num_modes):
    for author, group in tqdm(df.groupby('Author')):
        if author != UNKNOWN_AUTHOR:
            genres = [genre for genres in group['Book genres'] if
                      isinstance(genres, tuple) for genre in genres]
            if genres:
                mode_genres = [genre for genre, _ in Counter(genres)
                               .most_common(num_modes)]
                # We do this because normally can't assign tuple
                # and we get an error
                if len(df.loc[(df['Author'] == author) &
                       (df['Book genres'].isnull()), 'Book genres']) > 0:
                    for i in data.loc[(data['Author'] == 'Aaron Allston') &
                                      (data['Book genres'].isnull()),
                                      'Index'].values:
                        df.at[i, 'Book genres'] = tuple(mode_genres)

    return df


# Fill missing genres based on author with the 5 most common genres
data = fill_missing_genres(data, 5)

# Plot a histogram to find out median is better for 'Publication year' or mode
median = data['Publication year'].median()
mode = data['Publication year'].mode()[0]
mean = int(data['Publication year'].mean())

plt.hist(data['Publication year'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Data')

# Add markers for median and mode
plt.axvline(median, color='r', linestyle='dashed',
            linewidth=1, label=f'Median: {median}')
plt.axvline(mode, color='g', linestyle='dashed',
            linewidth=1, label=f'Mode: {mode}')
plt.axhline(mean, color='b', linestyle='dashed',
            linewidth=1, label=f'Mean: {mean}')
plt.xlim(1800, data['Publication year'].max())

plt.legend()
plt.show()

# Calculate the frequency of each value
value_counts = data['Publication year'].value_counts()

# Get the top 10 most frequent values
top_values = value_counts.head(10)

# Plot a bar plot for the top 10 most frequent values
plt.figure(figsize=(10, 6))
top_values.plot(kind='bar', color='skyblue')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Values')
plt.xticks(rotation=45)
plt.show()

# fill Publication Year and Publication date missing value
# with mode of data based on Top 10 values plot
mean_year = int(data['Publication year'].mode()[0])
data['Publication year'].fillna(mean_year, inplace=True)
data['Publication date'].fillna(mean_year, inplace=True)

# We can use "Title-Based" method or Pre-Trained NLP Model
# for Genre Prediction but in this step ignore this method
# and just set this missing value genre with UNKNOWN_GENRE variable


def fill_with_unknown_genre(x):
    return UNKNOWN_GENRE if pd.isnull(x) else x


data['Book genres'] = data['Book genres'].apply(fill_with_unknown_genre)

# Check the Nan value of all columns
print(data.isna().sum())

# save cleaned data
data.to_csv('cleaned_book_summary.csv', sep='\t',
            index=False, encoding='utf-8')

"""### EDA section"""

import ast
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from csv file
data = pd.read_csv('cleaned_book_summary.csv', sep='\t', encoding='utf-8')

# convert all genres values into tuple
for i in tqdm(range(len(data))):
    try:
        genres_tuple = ast.literal_eval(data.loc[i, 'Book genres'])
        if isinstance(genres_tuple, tuple):
            data.at[i, 'Book genres'] = genres_tuple
        else:
            data.at[i, 'Book genres'] = np.nan
    except (SyntaxError, ValueError):
        continue

def get_genres_distribution(df):
    """
    Calculate the distribution of book genres in the provided DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing a column 'Book genres'
    with lists of genres for each book.

    Returns:
    pandas.Series: A Series object containing the count of each genre across
    all books in descending order.

    The function iterates through the 'Book genres' column of the DataFrame
    and counts the occurrences of each genre.
    It then sorts the genres_counts dictionary by the count of each genre in
    descending order. The sorted dictionary is converted into a Pandas Series
    and returned.
    """
    genres_counts = dict()
    for i, genres in enumerate(df['Book genres']):
        for genre in genres:
            genres_counts[genre] = genres_counts.get(genre, 0) + 1

    # Sort the genres_counts dictionary by values in descending order
    sorted_genres_counts = dict(sorted(genres_counts.items(),
                                       key=lambda item: item[1],
                                       reverse=True))

    # Convert the sorted dictionary to a Pandas Series
    genre_series = pd.Series(sorted_genres_counts)

    return genre_series

# Get basic information about data
print(f"Number of books: {data.shape[0]}")
# Check for any remaining missing values
print(f"Missing values: {data.isnull().sum().any()}")

# Analyze summary length
summary_stats = data['Summary Length'].describe()
print(f"\nSummary length statistics:\n{summary_stats}")

# box plot for summary length
plt.figure(figsize=(15, 3))
# Creating the box plot
plt.boxplot(data['Summary Length'], vert=False)
plt.title('Summary Length Box Plot')
plt.xlabel('Summary Length')
plt.ylabel('Value')
plt.show()

# Analyze publication year
publication_stats = data['Publication year'].describe()
print(f"\nPublication year statistics:\n{publication_stats}")

# box plot for publication year
plt.figure(figsize=(15, 3))
# Creating the box plot
plt.boxplot(data['Publication year'], vert=False)
plt.title('Publication Year Box Plot')
plt.xlabel('Publication Year')
plt.ylabel('Value')
plt.show()

# Genre distribution
genre_counts = get_genres_distribution(data)
print(f"\nGenre distribution:\n{genre_counts}")

# Get the top 10 most frequent genres
top_genres = genre_counts.head(10)

# Plot genre distribution
plt.figure(figsize=(20, 6))
top_genres.plot(kind='bar', color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Number of Books')
plt.title('Top 10 Most Distribution of Book Genres')
plt.xticks(rotation=0)
plt.show()

# Publication year distribution
year_counts = data['Publication year'].value_counts()
print(f"\nPublication year distribution:\n\n{year_counts}")

# Get the top 20 most frequent publication year
top_years = year_counts.head(20)

# Plot publication year distribution
plt.figure(figsize=(20, 6))
top_years.plot(kind='bar', color='skyblue')
plt.xlabel("Publication year")
plt.ylabel("Number of Publication year")
plt.title("Top 20 Distribution of Publication year")
plt.xticks(rotation=45)
plt.show()

# Author analysis
author_counts = data['Author'].value_counts()
print(f"\nAuthor analysis:\n\n{author_counts}")

# Get the top 20 most frequent authors
top_authors = author_counts.head(21)[1:]  # remove the Unknown author

# Plot most frequent authors
plt.figure(figsize=(20, 6))
top_authors.plot(kind='bar', color='skyblue')
plt.xlabel("Authors")
plt.ylabel("Number of books")
plt.title("Author Popularity")
plt.xticks(rotation=45)
plt.show()

# Genre Co-occurrence analysis
genre_pairs = []
for genres in data['Book genres']:
    for i in range(len(genres) - 1):
        for j in range(i + 1, len(genres)):
            pair = (genres[i], genres[j])
            genre_pairs.append(pair)

# Count occurrences
genre_pair_counts = pd.Series(genre_pairs).value_counts()

# Print top 10 most frequent genre pairs
most_frequent_genre_pairs = genre_pair_counts.head(10)
print(most_frequent_genre_pairs)

# Plot most frequent genre co-occurrence
plt.figure(figsize=(20, 6))
most_frequent_genre_pairs.plot(kind='bar', color='skyblue')
plt.xlabel("Genre Pair")
plt.ylabel("Number of Occurrences")
plt.title("Genre Co-occurrence")
plt.xticks(rotation=45)
plt.show()

"""## Step 2: NLP Component"""

# import google drive to use gpu for this section
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers torch rouge-score

from tqdm import tqdm
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer, move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn',
                                          clean_up_tokenization_spaces=True)
model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn').to(device)

# Load dataset from csv file from drive for google colab
file_path = 'drive/MyDrive/cleaned_book_summary.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

# # Load dataset from local csv file
# data = pd.read_csv('cleaned_book_summary.csv', sep='\t', encoding='utf-8')

def summarize_batch(texts, max_input_length=1024, max_output_length=100, num_beams=2, device='cpu'):
    """
    Summarizes a batch of input texts using the BART model.

    Args:
    texts (list of str): List of original texts to be summarized.
    max_input_length (int): Maximum number of tokens in the input text.
    max_output_length (int): Maximum number of tokens in the summarized text.
    num_beams (int): Number of beams for beam search.
    device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    list of str: List of summarized texts.
    """
    # Tokenize and truncate the input to the max input length (batch processing)
    inputs = tokenizer(texts, return_tensors="pt", max_length=max_input_length, truncation=True, padding=True).to(device)

    # Generate the summary for the batch
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=num_beams,  # Beam search for diversity
        max_length=max_output_length,  # Limit the length of the output summary
        early_stopping=True  # Stop when the model is confident in its answer
    )

    # Decode the generated tokens into a string (summary)
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]

# Example usage on a single text summary
sample_text = data.loc[0, 'Cleaned Summary']
short_summary = summarize_batch([sample_text], device=device)[0]
print("Cleaned Summary lenght:", len(sample_text.split()))
print("Summarized Text lenght:", len(short_summary.split()))
print("Summarized Text:", short_summary)

# Check for accuracy of the model with sample text
from rouge_score import rouge_scorer

# Define the original text and summarized text
# we use sample_text as original_text and short_summary as summarized text

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Compute ROUGE scores
scores = scorer.score(sample_text, short_summary)

# Print ROUGE scores
print("ROUGE-1: ", scores['rouge1'])
print("ROUGE-2: ", scores['rouge2'])
print("ROUGE-L: ", scores['rougeL'])

"""The ROUGE scores you've obtained provide valuable insight into the performance of your summarization model. Here's a breakdown of the results:

### **ROUGE-1:**
- **Precision = 1.0**: This indicates that 100% of the unigrams (individual words) in the summary are found in the original text. In other words, all the words in the summary are present in the original.
- **Recall = 0.136 (approx)**: Only about 13.6% of the unigrams from the original text are included in the summary, meaning the summary is quite condensed.
- **F1-Score = 0.239 (approx)**: This score represents the harmonic mean between precision and recall, balancing both metrics. While precision is perfect, the relatively low recall pulls the F1-score down.

### **ROUGE-2:**
- **Precision = 1.0**: All the bigrams (pairs of consecutive words) in the summary appear in the original text.
- **Recall = 0.134 (approx)**: Only about 13.4% of the bigrams from the original text are present in the summary, indicating that the bigram coverage of the original content is limited.
- **F1-Score = 0.236 (approx)**: This score reflects the balance between precision and recall for bigrams. Similar to ROUGE-1, the perfect precision is offset by low recall.

### **ROUGE-L:**
- **Precision = 1.0**: All of the longest common subsequences (chunks of text that maintain word order) in the summary are present in the original text.
- **Recall = 0.136 (approx)**: About 13.6% of the original text's sequences are included in the summary, again showing that the summary is highly condensed.
- **F1-Score = 0.239 (approx)**: This score balances precision and recall for subsequences, similar to the unigrams and bigrams.

### **Interpretation:**
- **Perfect precision (1.0)**: All of the words, bigrams, and sequences in the summary are taken directly from the original text. This means the summary is highly faithful to the original in terms of word choice and order.
- **Low recall (~13.5%)**: The summary only includes a small portion of the original text, indicating a significant reduction in content. This could be because the summary is intentionally short.
- **F1-scores (~0.239)**: The F1-scores reflect the balance between precision and recall. Since precision is perfect but recall is low, the F1-scores are relatively low.

### **Summary of Findings:**
- The summarization model generates summaries that are **very precise**, ensuring that the words and phrases used are accurately drawn from the original text.
- However, the **low recall** suggests that the summaries omit a large portion of the original content, making them very condensed.
- If your goal is to produce **short, concise summaries**, this level of condensation may be ideal. On the other hand, if you'd like to retain more of the original content, you may need to **increase the summary length** or adjust the model's parameters.
"""

def batch_summarization(df, batch_size=32, num_beams=2, device='cpu'):
    """
    Summarizes the text in batches to manage memory and computation, leveraging GPU and batching.

    Args:
    df (DataFrame): The input DataFrame containing the 'Cleaned Summary' column.
    batch_size (int): Number of rows to process in each batch.
    num_beams (int): Number of beams for beam search.
    device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    DataFrame: The DataFrame with a new column 'Summarized_Text'.
    """
    # Ensure the new column exists
    df['Summarized_Text'] = ""

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        texts = batch_df['Cleaned Summary'].tolist()  # Extract the batch of texts
        summaries = summarize_batch(texts, num_beams=num_beams, device=device)  # Summarize the batch
        df.loc[batch_df.index, 'Summarized_Text'] = summaries  # Update the DataFrame with the summarized text

        print(f"Processed batch {i} to {i+batch_size}")

    return df

# Summarize the text in batches
data = batch_summarization(data, batch_size=32, num_beams=2, device=device)

# Display the first few summarized texts
data[['Cleaned Summary', 'Summarized_Text']].head()

# Save summarized book summary data to drive from google colab
file_path = 'drive/MyDrive/summarized_book_summary.csv'
data.to_csv(file_path, sep='\t',
            index=False, encoding='utf-8')

# # Save summarized book summary data to local system
# data.to_csv('summarized_book_summary.csv', sep='\t',
#             index=False, encoding='utf-8')

"""## Step 3: Computer Vision Component"""

# import google drive to use gpu for this section
from google.colab import drive
drive.mount('/content/drive')

# install the necessary libraries
!pip install -q diffusers transformers torch

# import libraries
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

# Load dataset from csv file from drive for google colab
file_path = 'drive/MyDrive/summarized_book_summary.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

# # Load dataset from local csv file
# data = pd.read_csv('summarized_book_summary.csv', sep='\t', encoding='utf-8')

# Check if a GPU is available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               torch_dtype=torch.float16)\
                                               .to(device)

# Ensure the output folder exists
output_folder = "drive/MyDrive/generated_images"
os.makedirs(output_folder, exist_ok=True)

def text_to_image(text, num_images_per_prompt=1, height=256, width=256,
                  pipe=None, device="cpu"):
    """
    Generates an image from the provided text using the Stable Diffusion
    pipeline.

    Args:
        text (str): The text description to convert into an image.
        num_images_per_prompt (int, optional): The number of images to generate
                                               per prompt. Defaults to 1.
        height (int, optional): The desired height of the generated image.
                                Defaults to 256.
        width (int, optional): The desired width of the generated image.
                               Defaults to 256.
        pipe (StableDiffusionPipeline, optional): The Stable Diffusion model
                                                  pipeline. If not provided, a
                                                  default model will be loaded.
        device (str, optional): The device to use for inference (e.g., 'cpu',
                                'cuda'). Defaults to 'cpu'.

    Returns:
        image (PIL Image): Generated image.
    """

    if pipe is None:
        # Load a default Stable Diffusion model if not provided
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)\
            .to(device)

    # Generate the image
    image = pipe(text, num_images_per_prompt=num_images_per_prompt,
                 height=height, width=width).images[0]

    return image

def batch_text_to_images(df, pipe, batch_size=32, start_index=0,
                         stop_index=None, num_images_per_prompt=1, height=256,
                         width=256, device="cpu",
                         output_folder="generated_images"):
    """
    Generates images in batches from a DataFrame containing summarized text and
    saves them to the specified output folder.

    Args:
        df (pandas.DataFrame): The DataFrame containing the summarized text in a
                               column named 'Summarized_Text'.
        pipe (StableDiffusionPipeline): The pre-trained Stable Diffusion model
                                        pipeline used for image generation.
        batch_size (int, optional): The number of rows to process in each batch.
                                    Defaults to 32.
        start_index (int, optional): The starting index for iterating through
                                     the DataFrame. Defaults to 0.
        stop_index (int, optional): The ending index for iterating through the
                                    DataFrame (exclusive). If not provided,
                                    iterates through the entire DataFrame.
        num_images_per_prompt (int, optional): The number of images to generate
                                                for each summarized text entry.
                                                Defaults to 1.
        height (int, optional): The desired height of the generated images.
                                Defaults to 256.
        width (int, optional): The desired width of the generated images.
                               Defaults to 256.
        device (str, optional): The device to use for inference (e.g., 'cpu',
                                'cuda'). Defaults to 'cpu'.
        output_folder (str, optional): The path to the output folder where
                                       generated images will be saved.
                                       Defaults to 'generated_images'.

    Raises:
        Exception: Any exception that occurs during image generation for a
                   particular index in the DataFrame will be printed.
    """
    if not stop_index:
      stop_index = len(df)

    # Loop through the dataframe in batches
    for i in tqdm(range(start_index, stop_index, batch_size),
                  desc="Generating Images"):
        batch_df = df.iloc[i:i+batch_size]
        for idx, row in batch_df.iterrows():
            try:
                # Generate the image from the summarized text
                img = text_to_image(row['Summarized_Text'],
                                    num_images_per_prompt=num_images_per_prompt,
                                    height=height, width=width, pipe=pipe,
                                    device=device)

                # Save the image with the index as the filename
                img.save(os.path.join(output_folder, f"{idx}_summary.png"))
            except Exception as e:
                print(f"Error generating image for index {idx}: {e}")

        print(f"Processed batch {i} to {i+batch_size}")

def display_image(image):
    """
    Display the generated image using matplotlib.

    Args:
    image (PIL Image): The image to display.
    """
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Example usage with a sample text
sample_text = data.loc[0, 'Summarized_Text']

# Generate image for sample text
sample_image = text_to_image(text, pipe=pipe, device=device)

# Display the generated image
display_image(sample_image)

# Example usage with your DataFrame
batch_text_to_images(data, pipe=pipe, device=device, stop_index=1000)