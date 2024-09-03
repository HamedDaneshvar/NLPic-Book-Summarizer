# Technical assessment

We are given the dataset `booksummaries.txt` and aim to create a Python script or notebook that combines NLP, Computer Vision, Machine Learning and Data Visualization to analyze and understand a dataset of book summaries. This includes cleaning the data, exploring its features through EDA, shortening the summaries and converting them to images, which we describe in detail below. And we want to get " The condensed text summaries", "The converted images" and " The findings from the exploratory data analysis (EDA)" by doing these steps.

## Data Preprocessing and EDA

### Load and preprocessing data

- Loading the data using the pandas library and adding a header to it.
- Display head of the data and shape of it.
- Removing the columns `Wikipedia article ID` and `Freebase ID` from the data.
- Adding an index to the dataset.
- Cleaning up the genres of the books and changing their structure to a list of genres.
- Adding the `Publication year` column based on the `Publication date` column, which only contains the year of publication of the book.
- Identifying duplicate rows and removing them based on the `Book title` and `Author` columns.
- Cleaning the `Plot summary` column by converting to lowercase, removing punctuation marks, digits, extra spaces, special characters, stop words, and finally trimming leading and trailing spaces. Adding this cleaned content as a new column named `Cleaned Summary`.
- Removing outliers using the IQR method.
- Filling missing data in the `Author` column with the value `Unknown` as we cannot assign a book to a different author.
- Filling missing data in the `Book genres` based on the `Author` and other genres of books by this author that are available. Identifying the 5 common genres of books by this author.
- Calculating the median, mode, and mean of the `Publication year` and plotting a histogram to determine which of these measures is suitable for the missing `Publication Year` and `Publication date` data.

<img src='./assets/publication_year_hist.png'></img>

- Due to the overlap and abundance of data, it was not informative to plot a histogram. Therefore, we plot the histogram for the Top 10 years to see that all of these top 10 years belong to the period after 2000. Hence, we use the mode as a value to fill in the missing `Publication Year` and `Publication date` data.

<img src='./assets/top10_publication_year_hist.png'></img>

- Finally, we assign the value `Unknown` for the remaining missing data in the `Book genres`.
- Lastly, we verify that there are no missing data left for any of the columns.

There are multiple methods for handling missing data, one of which we utilized for the `Book genres` column. This method involved inferring the genres of books based on other works by the author, identifying 5 common genres from these other works. However, there are other methods that we can employ, and we will mention three that are relevant to this issue. 

The first method we cannot use is the `Forwardfill/Backwardfill` method, which can fill in missing data if the data exhibits logical sequences. Another method involves utilizing external sources to find data, such as using external datasets, APIs (like Google Books API, GoodReads, or Freebase), or by writing a web scraper to gather the required information (apparently, the values in the `Wikipedia article ID` column are not valid). 

The third method involves using a pre-trained NLP model that can specifically identify book genres according to our needs.
