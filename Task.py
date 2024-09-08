#!/usr/bin/env python
# coding: utf-8

# # DataCoLab Task Assignment

# ## Step 1: Data Preprocessing and EDA


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
header = ['Wikipedia article ID', 'Freebase ID', 'Book title', 'Author',
          'Publication date', 'Book genres (Freebase ID:name tuples)',
          'Plot summary']

# Load dataset from text file
data = pd.read_csv('booksummaries.txt', sep='\t', header=None,
                   names=header, encoding='utf-8')

# Display sample of data
data.head()

# Display shape of data
print(f"Shape of dataset: {data.shape}")

"""### Preprocess Section"""

# # Drop "Freebase ID" column cause that values are unique and useless
data = data.drop("Wikipedia article ID", axis=1)
data = data.drop("Freebase ID", axis=1)

# Add "Index" column by reset it
data = data.reset_index()
data.rename(columns={'index': 'Index'}, inplace=True)

# convert all genres values into tuple
for i in tqdm(range(len(data))):
    try:
        genres_dict = json.loads(
            data.loc[i, 'Book genres (Freebase ID:name tuples)'])
        if isinstance(genres_dict, dict):
            data.at[i, 'Book genres (Freebase ID:name tuples)'] = tuple(
                genres_dict.values())
        else:
            data.at[i, 'Book genres (Freebase ID:name tuples)'] = np.nan
    # Handle NaN values and JSON decoding errors
    except (TypeError, json.JSONDecodeError):
        continue

data.rename(columns={'Book genres (Freebase ID:name tuples)': 'Book genres'},
            inplace=True)

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


data['Cleaned Summary'] = data['Plot summary'].apply(clean_text)

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