# NLPic Book Summarizer

## Overview
This project involves the analysis and understanding of a dataset of book summaries using NLP, Computer Vision, Machine Learning, and Data Visualization techniques. 

For the Computer Vision component, due to limitations in Google Colab, 1024 random text summaries have been selected and their images have been generated.

## Objective
Create a Python script or notebook that combines the mentioned technologies to perform the following tasks:
- Data Preprocessing and EDA
- NLP Component
- Computer Vision Component

## Requirements
1. **Data Preprocessing and EDA:**
   - Clean the data by handling missing values.
   - Explore the dataset to understand its characteristics.

2. **NLP Component:**
   - Condense the book summaries to make them shorter.

3. **Computer Vision Component:**
   - Convert the condensed text summaries into images using a Text-to-Image model.

## Project Structure
- `docs/`: Directory containing project documentation.
  - [English Documentation](./docs/README.md)
  - [Persian Documentation](./docs/README-Fa.md)
- `checkpoint_datasets/`: Directory containing datasets created after each step.
- `assets/`: Directory containing project images and generated images.
- `booksummaries.txt`: Raw dataset.
- `final_book_summaries.csv`: Final processed dataset and requested dataset.
- `Task.ipynb` or `Task.py`: Source code file.
- `requirements.txt`: File listing the project dependencies.
- `.env.sample`: Sample .env file for Google Book API key.

## Setting up Google Book API
1. Obtain a Google Book API key.
2. Create a `.env` file based on the provided `.env.sample`.
3. Add your Google Book API key to the `.env` file.

## Generated Images
For the generated images directory, refer to this [Link](./assets/generated_images/) to access the images created during the project.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Guide

For the EDA section:
- [English EDA Part](./docs/Readme.md#understanding-and-exploring-dataset-features-through-eda)
- [بخش تحلیل داده‌های اکتشافی فارسی](./docs/Readme-Fa.md#تحلیل-داده-های-اکتشافی-eda)

For detailed guides:
- [English Guide](./docs/README.md)
- [راهنمای فارسی](./docs/README_Fa.md)
