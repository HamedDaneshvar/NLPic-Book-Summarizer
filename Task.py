#!/usr/bin/env python
# coding: utf-8

# # DataCoLab Task Assignment

## Step 1: Data Preprocessing and EDA

# import google drive to use gpu for this section
from google.colab import drive
drive.mount('/content/drive')

!pip install -q python-dotenv

# Import libraries
import os
import re
import json
import time
import string
from tqdm import tqdm
from collections import Counter
import requests
from urllib.parse import urlencode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CONSTANTS
UNKNOWN_GENRE = ('Unknown',)
UNKNOWN_AUTHOR = 'Unknown'
UNKNOWN_PUBLICATION_DATE = 'Unknown'

"""#### load env variable"""

_ = load_dotenv(find_dotenv())

"""### Load dataset from google drive"""

# header for dataset
header = ['Wikipedia_Article_ID', 'Freebase_ID', 'Book_Title', 'Author',
          'Publication_Date', 'Book_Genres_(Freebase_ID:name_tuples)',
          'Plot_Summary']

# Load dataset from text file in google drive
file_path = 'drive/MyDrive/booksummaries.txt'
data = pd.read_csv(file_path, sep='\t', header=None,
                   names=header, encoding='utf-8')

"""### Load dataset from local system"""

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
# data = data.drop("Wikipedia_Article_ID", axis=1)
# data = data.drop("Freebase_ID", axis=1)

# Add "Index" column by reset it
data = data.reset_index()
data.rename(columns={'index': 'Index'}, inplace=True)

# convert all genres values into tuple and remove Freebase_ID
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

# Check the Nan value of all columns
print(data.isna().sum())

"""### Fill Author and Publication date missing values with Google Book API"""

def clean_title(title):
    """
    Clean and sanitize a title for encoding in a URL query parameter.

    Parameters:
    title (str): The title to be cleaned and formatted.

    Returns:
    str: The cleaned and formatted title ready for encoding.

    The function performs the following steps to clean the title:
    1. Removes leading and trailing whitespaces.
    2. Removes special characters except for letters, digits, spaces, and specified punctuation signs.
    3. Removes extra whitespaces.

    Example:
    >>> title = "The //!@#Great Gatsby's   "
    >>> cleaned_title = clean_title(title)
    >>> print(cleaned_title)
    'The Great Gatsby's'
    """
    title = title.strip()
    title = ''.join(char if char.isalnum() or char.isspace() or char in
        ["'", "-", ",", ":", ";"] else ' ' for char in title)
    title = ' '.join(title.split())

    return title

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

    title = clean_title(title)

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
for index, row in tqdm(data.iterrows()):
    # Check if either Author or Publication date is missing
    if pd.isnull(row['Author']) or pd.isnull(row['Publication_Date']):
        # Use the existing author if available, otherwise send None
        author = row['Author'] if not pd.isnull(row['Author']) else None
        book_info = get_book_info(row['Book_Title'], author)
        if book_info:
            data.at[index, 'Author'] = book_info['author']
            data.at[index, 'Publication_Date'] = book_info['publication_date']
        time.sleep(1)

# Check the Nan value of 'Author' and 'Publication_Date' columns
print(data.isna().sum())

# Checkpoint 1 for to save data into drive
file_path = 'drive/MyDrive/checkpoint1_book_summaries.csv'
data.to_csv(file_path, sep='\t', index=False, encoding='utf-8')

file_path = 'drive/MyDrive/checkpoint1_book_summaries.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

def extract_year(pub_date):
    """
    Convert publication date to just the year
    (if it's in YYYY-MM-DD format)
    """
    if isinstance(pub_date, str) and '-' in pub_date:
        return pd.to_datetime(pub_date, errors='coerce').year
    elif isinstance(pub_date, str) and not pub_date.isdigit():
        if year_match := re.search(r'\d{4}', pub_date):
            return int(year_match.group())
        else:
            return np.nan
    return int(pub_date) if isinstance(pub_date, str) and pub_date.isdigit()\
        else np.nan

data['Publication_Year'] = data['Publication_Date'].apply(extract_year)

# Remove duplicate rows
# Check for duplicates based on a unique identifier or key columns
duplicates = data.duplicated(subset=['Book_Title', 'Author'], keep='first')
print(f"Number of duplicate rows: {duplicates.sum()}")

# # Drop duplicates
# data = data.drop_duplicates(subset=['Book_Title', 'Author'], keep='first')

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

print(f"Number of null Cleaned Summary rows:",
      len(data[data['Cleaned_Summary'].isnull()]))

# Drop rows where 'Cleaned_Summary' is NaN
data = data.dropna(subset=['Cleaned_Summary'])

# summary length
data['Summary_Length'] = data['Cleaned_Summary']\
    .apply(lambda x: len(x.split()))

# Detect outliers in Summary_Length
Q1 = data['Summary_Length'].quantile(0.25)
Q3 = data['Summary_Length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['Summary_Length'] < lower_bound) |
                (data['Summary_Length'] > upper_bound)]
print(f"Number of outliers in Summary Length: {outliers.shape[0]}")

# # Remove outliers
# data = data[(data['Summary Length'] >= lower_bound) &
#             (data['Summary Length'] <= upper_bound)]

# Fill Author missing values with UNKNOWN_AUTHOR variable because
# We cannot put anyone's book under the name of another author
data['Author'].fillna(UNKNOWN_AUTHOR, inplace=True)

# Fill missing values book generes based of other author's book
# this section of code is not good and fix just 2 rows:)))


def fill_missing_genres(df, num_modes):
    for author, group in tqdm(df.groupby('Author')):
        if author != UNKNOWN_AUTHOR:
            genres = [genre for genres in group['Book_Genres'] if
                      isinstance(genres, tuple) for genre in genres]
            if genres:
                mode_genres = [genre for genre, _ in Counter(genres)
                               .most_common(num_modes)]
                # We do this because normally can't assign tuple
                # and we get an error
                if len(df.loc[(df['Author'] == author) &
                       (df['Book_Genres'].isnull()), 'Book_Genres']) > 0:
                    for i in data.loc[(data['Author'] == author) &
                                      (data['Book_Genres'].isnull()),
                                      'Index'].values:
                        df.at[i, 'Book_Genres'] = tuple(mode_genres)

    return df


# Fill missing genres based on author with the 5 most common genres
data = fill_missing_genres(data, 5)

# Check the Nan value of 'Author' and 'Publication_Date' columns
print(data.isna().sum())

# Plot a histogram to find out median is better for 'Publication_Year' or mode
median = data['Publication_Year'].median()
mode = data['Publication_Year'].mode()[0]
mean = int(data['Publication_Year'].mean())

plt.hist(data['Publication_Year'], bins=10, color='skyblue', edgecolor='black')
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
plt.xlim(1800, data['Publication_Year'].max())

plt.legend()
plt.show()

# Calculate the frequency of each value
value_counts = data['Publication_Year'].value_counts()

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
mean_year = int(data['Publication_Year'].mode()[0])
data['Publication_Year'].fillna(mean_year, inplace=True)
data['Publication_Date'].fillna(mean_year, inplace=True)

# We can use "Title-Based" method or Pre-Trained NLP Model
# for Genre Prediction but in this step ignore this method
# and just set this missing value genre with UNKNOWN_GENRE variable


def fill_with_unknown_genre(x):
    return UNKNOWN_GENRE if pd.isnull(x) else x


data['Book_Genres'] = data['Book_Genres'].apply(fill_with_unknown_genre)

# Check the Nan value of all columns
print(data.isna().sum())

# Save cleaned data after preprocessing
file_path = 'drive/MyDrive/cleaned_book_summaries.csv'
data.to_csv(file_path, sep='\t', index=False, encoding='utf-8')

"""## Step 2: NLP Component"""

# import google drive to use gpu for this section
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers torch rouge-score

from tqdm import tqdm
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer

# Load pre-trained BART model and tokenizer, move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn',
                                          clean_up_tokenization_spaces=True)
model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn').to(device)

# Load dataset from csv file from drive for google colab
file_path = 'drive/MyDrive/cleaned_book_summaries.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

# # Load dataset from local csv file
# data = pd.read_csv('cleaned_book_summaries.csv', sep='\t', encoding='utf-8')

def summarize_batch(texts, max_input_length=1024, max_output_length=75, num_beams=2, device='cpu'):
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

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Example usage on a single text summary
sample_text = data.loc[12390, 'Cleaned_Summary']  # longest text in dataset
short_summary = summarize_batch([sample_text], device=device)[0]
print("Cleaned Summary lenght:", len(sample_text.split()))
print("Summarized Text lenght:", len(short_summary.split()))
print("Summarized Text:", short_summary)

# Compute ROUGE scores
scores = scorer.score(sample_text, short_summary)

# Print ROUGE scores
print("ROUGE-1: ", scores['rouge1'])
print("ROUGE-2: ", scores['rouge2'])
print("ROUGE-L: ", scores['rougeL'])

"""The ROUGE scores obtained for the longest text in the dataset provide us with valuable insights into the performance of summarization model. Below is an analysis of these results:

### **ROUGE-1:**
- **Precision = 1.0**: This indicates that 100% of the unigrams (individual words) in the summary are present in the original text. The summary is exclusively composed of words from the original.
- **Recall = 0.0049**: Only about 0.49% of the unigrams from the original text are included in the summary. This very low recall suggests the summary is extremely condensed.
- **F1-Score = 0.0098**: The harmonic mean between precision and recall is low due to the significant gap between the two. While the model captures every word it chooses with perfect precision, it includes very few words from the original.

### **ROUGE-2:**
- **Precision = 0.62**: This means 62% of the bigrams (pairs of consecutive words) in the summary are present in the original text. However, this isn't as high as the unigram precision, indicating that while individual words are captured well, their order might not be as well preserved.
- **Recall = 0.0030**: About 0.3% of the bigrams from the original text are present in the summary, which suggests that only a very tiny fraction of the original bigram content is captured.
- **F1-Score = 0.0060**: The F1 score is quite low due to the large gap between precision and recall, indicating that the summary may be missing important contextual sequences of words.

### **ROUGE-L:**
- **Precision = 0.824**: Around 82.4% of the longest common subsequences (LCS) between the summary and original text are preserved, which suggests the model does a reasonable job of maintaining word order for the parts of the original text it includes.
- **Recall = 0.0041**: Only 0.41% of the sequences from the original text appear in the summary, indicating that while the word order is preserved, a very small amount of content is actually included.
- **F1-Score = 0.0081**: The low F1 score again reflects the tension between high precision and low recall, meaning the model extracts sequences faithfully but covers only a tiny portion of the original text.

### **Interpretation:**
- **High precision, low recall**: Across all the ROUGE scores, precision is relatively high (especially for ROUGE-1), but recall is extremely low. This suggests that the model creates **very short summaries**, where most of the content is omitted, but what is included is perfectly selected.
- **Condensed summary**: Given the very low recall (~0.49% for ROUGE-1 and ~0.3% for ROUGE-2), it seems the summary is heavily reduced in length and content compared to the original text.
- **Bigram and sequence preservation**: While bigram precision and LCS precision are somewhat decent, the low recall shows that the summary may not be capturing much of the original context or detail.

### **Summary of Findings:**
- Your summarization model is **highly precise** in choosing words and maintaining word order from the original text, but it summarizes the text in a very condensed form, leading to a **low recall**.
- If you're aiming for **concise summaries**, this may be acceptable, but the summaries may be missing a lot of important content.
- If our goal is to generate **short, accurate summaries**, this level of compression may be ideal, aligning with our objective. On the other hand, if we aim to preserve more content from the original material, we might need to **increase the summary length** (adjusting `max_output_length`) or adjust the model parameters.
"""

# Example usage on a single text summary
sample_text = data.loc[0, 'Cleaned_Summary']
short_summary = summarize_batch([sample_text], device=device)[0]
print("Cleaned Summary lenght:", len(sample_text.split()))
print("Summarized Text lenght:", len(short_summary.split()))
print("Summarized Text:", short_summary)

# Compute ROUGE scores
scores = scorer.score(sample_text, short_summary)

# Print ROUGE scores
print("ROUGE-1: ", scores['rouge1'])
print("ROUGE-2: ", scores['rouge2'])
print("ROUGE-L: ", scores['rougeL'])

"""The obtained ROUGE scores provide some useful insights into the performance of the summarization model. Below is an analysis based on the provided scores:

### **ROUGE-1:**
- **Precision = 1.0**: This means 100% of the unigrams (individual words) in the summary are found in the original text. The summary doesn’t introduce any new words, only using words from the original.
- **Recall = 0.0689**: About 6.89% of the unigrams from the original text are present in the summary. This indicates that only a small portion of the original content is captured in the summary.
- **F1-Score = 0.1288**: The F1 score, which is the harmonic mean between precision and recall, reflects a balance between the two. While the precision is perfect, the low recall keeps the F1 score down.

### **ROUGE-2:**
- **Precision = 1.0**: All of the bigrams (pairs of consecutive words) in the summary are directly taken from the original text, so there is no deviation from the word order or word choice.
- **Recall = 0.0679**: Only about 6.79% of the bigrams from the original text are captured in the summary, meaning that very few sequences of two consecutive words are included.
- **F1-Score = 0.1271**: This score reflects the balance between the perfect precision and the low recall, showing that while the summary is accurate in terms of bigram usage, it captures only a small part of the original.

### **ROUGE-L:**
- **Precision = 1.0**: The longest common subsequences (LCS) in the summary are perfectly matched to those in the original text, so word order is faithfully preserved.
- **Recall = 0.0689**: Similar to ROUGE-1, only 6.89% of the sequences from the original text are included in the summary, highlighting how much of the original content is omitted.
- **F1-Score = 0.1288**: The F1 score indicates a decent balance between precision and recall, where precision is perfect but recall is quite low.

### **Interpretation:**
- **Perfect precision (1.0)**: The summary is highly faithful to the original text, as every word, bigram, and sequence in the summary appears in the original. There is no extraneous or erroneous content.
- **Low recall (~6.9%)**: The summary is very condensed, including less than 7% of the original content. This suggests that while the summary is accurate, it covers only a small portion of the original text.
- **F1-scores (~0.128)**: The F1-scores reflect the model’s high precision but low recall, meaning it does a good job of capturing the words and phrases it chooses but doesn’t include much of the original content.

### **Summary of Findings:**
- The model generates summaries that are **extremely precise**, ensuring that all words and sequences in the summary are drawn from the original text.
- However, the **low recall** indicates that the summaries omit a significant portion of the original content, making them highly condensed.
- If our goal is to generate **short, accurate summaries**, this level of compression may be ideal, aligning with our objective. On the other hand, if we aim to preserve more content from the original material, we might need to **increase the summary length** (adjusting `max_output_length`) or adjust the model parameters.
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
        texts = batch_df['Cleaned_Summary'].tolist()  # Extract the batch of texts
        summaries = summarize_batch(texts, num_beams=num_beams, device=device)  # Summarize the batch
        df.loc[batch_df.index, 'Summarized_Text'] = summaries  # Update the DataFrame with the summarized text

        print(f"Processed batch {i} to {i+batch_size}")

    return df

# Summarize the text in batches
data = batch_summarization(data, batch_size=32, num_beams=2, device=device)

# Display the first few summarized texts
data[['Cleaned_Summary', 'Summarized_Text']].head()

# Save summarized book summary data to drive from google colab
file_path = 'drive/MyDrive/summarized_book_summary.csv'
data.to_csv(file_path, sep='\t',
            index=False, encoding='utf-8')

# # Save summarized book summary data to local system
# data.to_csv('summarized_book_summary.csv', sep='\t',
#             index=False, encoding='utf-8')

"""## Step 3 Predicted Book Genres with Bert NLP model"""

# Get the distinct book genres
unique_genres = dict()
for index, row in data.iterrows():
    if pd.isnull(row['Book_Genres']):
        continue
    for genre in row['Book_Genres']:
        unique_genres[genre] = unique_genres.get(genre, 0) + 1

total_genres = [k for k, v in sorted(unique_genres.items(), key=lambda x: x[1], reverse=True)]

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_genres))

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=len(total_genres))

data.head(15)

# Preprocess plot summary
text = data.loc[7, 'Plot_Summary']
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# Map predicted index to genre
predicted_genre = total_genres[predictions.item()]
print("Predicted genre:", predicted_genre)

"""## Step 4: Computer Vision Component"""

# import google drive to use gpu for this section
from google.colab import drive
drive.mount('/content/drive')

# install the necessary libraries
!pip install -q diffusers transformers torch

# import libraries
import os
import shutil
import zipfile
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
output_folder = "./generated_images"
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

def zip_directory(directory_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(directory_path))
                zipf.write(file_path, arcname)

sample_text = data.loc[0, 'Summarized_Text']

# Generate image for sample text
sample_image = text_to_image(text, pipe=pipe, device=device)

# Display the generated image
display_image(sample_image)

# Generate images for data 0 to 1000
batch_text_to_images(data, pipe=pipe, device=device, stop_index=1000,
                     output_folder=output_folder)

# Ensure the output folder exists
output_folder = "./generated_images_00"
os.makedirs(output_folder, exist_ok=True)
zip_file_path = "./generated_images_00.zip"
zip_directory(output_folder, zip_file_path)
shutil.copy(zip_file_path, "drive/MyDrive/generated_images_00.zip")

# Generate images for data 1000 to 2000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=1000, stop_index=2000,
                     output_folder=output_folder)

# Ensure the output folder exists
output_folder = "./generated_images_01"
os.makedirs(output_folder, exist_ok=True)
zip_file_path = "./generated_images_01.zip"
zip_directory(output_folder, zip_file_path)
shutil.copy(zip_file_path, "drive/MyDrive/generated_images_01.zip")

# Generate images for data 2000 to 3000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=2000, stop_index=3000,
                     output_folder=output_folder)

# Ensure the output folder exists
output_folder = "./generated_images_02"
os.makedirs(output_folder, exist_ok=True)
zip_file_path = "./generated_images_02.zip"
zip_directory(output_folder, zip_file_path)
shutil.copy(zip_file_path, "drive/MyDrive/generated_images_02.zip")

# Generate images for data 3000 to 4000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=3000, stop_index=4000,
                     output_folder=output_folder)

# Generate images for data 4000 to 5000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=4000, stop_index=5000,
                     output_folder=output_folder)

# Generate images for data 5000 to 6000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=5000, stop_index=6000,
                     output_folder=output_folder)

# Generate images for data 6000 to 7000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=6000, stop_index=7000,
                     output_folder=output_folder)

# Generate images for data 7000 to 8000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=7000, stop_index=8000,
                     output_folder=output_folder)

# Generate images for data 8000 to 9000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=8000, stop_index=9000,
                     output_folder=output_folder)

# Generate images for data 9000 to 10000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=9000, stop_index=10000,
                     output_folder=output_folder)

# Generate images for data 10000 to 11000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=10000, stop_index=11000,
                     output_folder=output_folder)

# Generate images for data 11000 to 12000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=11000, stop_index=12000,
                     output_folder=output_folder)

# Generate images for data 12000 to 13000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=12000, stop_index=13000,
                     output_folder=output_folder)

# Generate images for data 13000 to 14000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=13000, stop_index=14000,
                     output_folder=output_folder)

# Generate images for data 14000 to 15000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=14000, stop_index=15000,
                     output_folder=output_folder)

# Generate images for data 15000 to 16000
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=15000, stop_index=16000,
                     output_folder=output_folder)

# Generate images for data 16000 to end of dataset
batch_text_to_images(data, pipe=pipe, device=device,
                     start_index=16000,
                     output_folder=output_folder)

"""## Step 5: EDA section"""

import ast
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from csv file
data = pd.read_csv('cleaned_book_summaries.csv', sep='\t', encoding='utf-8')

# Convert all genres values into tuple
for i in tqdm(range(len(data))):
    try:
        genres_tuple = ast.literal_eval(data.loc[i, 'Book_Genres'])
        if isinstance(genres_tuple, tuple):
            data.at[i, 'Book_Genres'] = genres_tuple
        else:
            data.at[i, 'Book_Genres'] = np.nan
    except (SyntaxError, ValueError):
        continue

def get_genres_distribution(df):
    """
    Calculate the distribution of book genres in the provided DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing a column 'Book_Genres'
    with lists of genres for each book.

    Returns:
    pandas.Series: A Series object containing the count of each genre across
    all books in descending order.

    The function iterates through the 'Book_Genres' column of the DataFrame
    and counts the occurrences of each genre.
    It then sorts the genres_counts dictionary by the count of each genre in
    descending order. The sorted dictionary is converted into a Pandas Series
    and returned.
    """
    genres_counts = dict()
    for i, genres in enumerate(df['Book_Genres']):
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
summary_stats = data['Summary_Length'].describe()
print(f"\nSummary length statistics:\n{summary_stats}")

# box plot for summary length
plt.figure(figsize=(15, 3))
# Creating the box plot
plt.boxplot(data['Summary_Length'], vert=False)
plt.title('Summary Length Box Plot')
plt.xlabel('Summary Length')
plt.ylabel('Value')
plt.show()

# Analyze publication year
publication_stats = data['Publication_Year'].describe()
print(f"\nPublication year statistics:\n{publication_stats}")

# box plot for publication year
plt.figure(figsize=(15, 3))
# Creating the box plot
plt.boxplot(data['Publication_Year'], vert=False)
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
year_counts = data['Publication_Year'].value_counts()
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
for genres in data['Book_Genres']:
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