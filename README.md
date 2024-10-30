# Data Preprocessing and Feature Extraction

## Description
This project focuses on preparing and cleaning text data to enable effective analysis and modeling. Key tasks include reading text data, cleaning it for noise reduction, and performing feature extraction using techniques like TF-IDF and noun/subject identification. The end result is a dataset ready for exploration, modeling, and deeper analysis.

## Contents

The following steps outline the main components of this project:

### Data Loading and Reading:
- **Functionality**: Reads `.xml` text files from a specified directory.
- **Storage**: Stores each file's content in a list for further processing.

### Data Cleaning:
- **Non-ASCII Character Removal**: Remove characters that aren’t part of the ASCII standard.
- **Lowercasing**: Normalize text by converting to lowercase.
- **Stop Word Removal**: Eliminate common stop words to reduce noise.
- **Removing Punctuation and Numbers**: Remove unwanted characters for a cleaner text dataset.
- **Contraction Expansion**: Expand contractions (e.g., "don't" to "do not") for consistency.
- **Lemmatization and Stemming**: Normalize words to their base or root form.

### Feature Extraction:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Identify the most important words within each document.
- **Noun and Subject Extraction**: Extract nouns and subjects to focus on key topics within the text data.

## Code Structure

This repository includes the following main code blocks:

- `read_text_files(folder_path)`: Reads all `.xml` files in the folder and returns the content as a list.
- `remove_non_ascii(text)`: Removes non-ASCII characters.
- `convert_to_lowercase(text)`: Converts text to lowercase.
- `remove_noncontext_words(text)`: Removes specific non-contextual words from text data.
- `remove_stop_words(text)`: Removes common stop words.
- `remove_punctuation_and_newlines(text)`: Strips punctuation and unnecessary newlines.
- `lemmatize_text(text)`: Lemmatises words in the text.
- `stem_text(text)`: Applies stemming to words.
- `replace_contractions(text)`: Expands contractions.
- `find_most_important_word_TFIDF(texts)`: Applies TF-IDF to identify key words.
- `extract_nouns(text)`: Identifies and counts nouns.
- `extract_subjects(text)`: Identifies and counts subjects within clauses.

## Prerequisites

Make sure to install the required Python libraries:

```bash
pip install nltk pandas sklearn gensim
```

# Data Preprocessing and Feature Extraction: Building a Strong Foundation for Text Analysis

## Introduction

In the world of data science, effective data preprocessing and feature extraction are essential for meaningful analysis and modeling. This project, **Data Preprocessing and Feature Extraction**, focuses on preparing raw text data to facilitate a streamlined analysis process. By cleaning, standardizing, and extracting features, we ensure that the dataset is ready for advanced exploration and modeling.

The main goal of this project is to transform unstructured text data into a structured format. We’ll perform multiple preprocessing steps to clean the data, followed by feature extraction techniques, such as TF-IDF (Term Frequency-Inverse Document Frequency) and noun/subject identification, to derive insights from the text.

---

## Project Overview

### Objectives

The key objectives of this project are:
1. **Load and clean text data** to remove noise and irrelevant information.
2. **Normalize the text data** using tokenization, lemmatization, and removal of non-contextual words.
3. **Extract features** like important words and topics through TF-IDF and noun/subject identification.

This process prepares the data for downstream tasks, such as sentiment analysis, topic modeling, and further exploration.

### Dataset

The dataset consists of multiple XML files with text content. Each file represents a piece of text data to be cleaned and processed, providing a foundation for the feature extraction steps.

---

## Steps for Data Preprocessing and Feature Extraction

Here’s a step-by-step guide to our data preprocessing and feature extraction pipeline.

### Step 1: Data Loading

First, we need to read the XML files from a specified directory and store each file's content for processing.

```python
import os

# Function to read .xml files in folder_path and return their content in a list
def read_text_files(folder_path):
    blog_content = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.xml'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                blog_content.append(content)
    return blog_content
```
### Step 2: Data Cleaning
Raw text data often contains irrelevant characters, symbols, and structures that need to be removed. Our data cleaning steps include:

 - Non-ASCII Character Removal: Eliminates characters that aren’t part of the ASCII standard.
 - Lowercasing: Converts text to lowercase for uniformity.
 - Stop Word Removal: Removes common words that don’t add value to analysis.
 - Punctuation Removal: Strips punctuation for a cleaner dataset.
 - Contraction Expansion: Converts contractions (e.g., "don't" to "do not") for consistency.
 - Lemmatization and Stemming: Reduces words to their base forms, ensuring consistency across the dataset.

```import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

# Convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()

# Remove stop words
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

# Lemmatize and stem text
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize_text(text):
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def stem_text(text):
    words = text.split()
    return ' '.join([stemmer.stem(word) for word in words])
```
By applying these functions, we create a clean, standardized version of the text data.

### Step 3: Feature Extraction
Once the text is clean, we perform feature extraction to identify important words, topics, and entities within the text data.

3.1 TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF helps us identify the most important words within each document relative to the entire corpus, highlighting terms that carry specific importance.

```from sklearn.feature_extraction.text import TfidfVectorizer

# Apply TF-IDF to extract important words
tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
tfidf_matrix = tfidf.fit_transform(text_data)
```
from sklearn.feature_extraction.text import TfidfVectorizer

3.2 Noun and Subject Identification
Identifying nouns and subjects within sentences helps us focus on key entities and themes. We use part-of-speech tagging and lemmatization to retrieve nouns and subjects.

```from nltk import pos_tag, word_tokenize
from collections import Counter

# Extract nouns
def extract_nouns(text):
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
    return Counter(nouns)

# Extract subjects of clauses
def extract_subjects(text):
    sentences = nltk.sent_tokenize(text)
    subjects = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        for i in range(len(tagged_words)):
            word, pos = tagged_words[i]
            if pos.startswith('V') and i > 0:
                prev_word, prev_pos = tagged_words[i - 1]
                if prev_pos.startswith('N') or prev_pos == 'PRP':
                    subjects.append(prev_word)
    return Counter(subjects)
```
These functions allow us to focus on the most relevant nouns and subjects, creating a basis for further analysis of the text content.

## Results and Insights
The output of the data preprocessing and feature extraction steps provides us with a clean, structured dataset ready for analysis. Key elements extracted from the text data include:

 - Important Words: Using TF-IDF, we identify key terms that appear frequently within individual texts but not across the entire corpus.
 - Nouns and Subjects: The extracted nouns and subjects represent the core themes within each text.

## Code Structure
This project includes several core functions organized in a structured manner:

- Data Loading: read_text_files(folder_path) reads all XML files and returns their content.
- Data Cleaning: Multiple functions clean the text data, such as remove_non_ascii(text), convert_to_lowercase(text), and remove_stop_words(text).
- Feature Extraction: find_most_important_word_TFIDF(texts) for TF-IDF, and extract_nouns(text) and extract_subjects(text) for noun and subject identification.

## Sample Code Usage
Below is an example of how to load, clean, and extract features from the text data:
```# Load data
folder_path = 'path_to_data_folder'
text_data = read_text_files(folder_path)

# Clean data
cleaned_text_data = [remove_stop_words(convert_to_lowercase(remove_non_ascii(text))) for text in text_data]

# Extract features
tfidf_matrix = tfidf.fit_transform(cleaned_text_data)
nouns = [extract_nouns(text) for text in cleaned_text_data]
subjects = [extract_subjects(text) for text in cleaned_text_data]
```

## Conclusion
Data preprocessing and feature extraction lay the groundwork for further analysis. By cleaning the data and extracting meaningful features, we can efficiently move to exploratory analysis, topic modeling, sentiment analysis, and predictive modeling.

## Key Takeaways
- Essential Foundation: Preprocessing is crucial for ensuring reliable results in text analysis.
- Scalability: This pipeline can be scaled to larger datasets for more comprehensive projects.
- Versatility: Feature extraction techniques like TF-IDF and noun extraction are adaptable for various NLP applications.

This project highlights the power of data preprocessing and feature extraction, turning unstructured text into a structured dataset ready for insightful analysis.

## Next Steps
With a clean and structured dataset, future projects could explore:

- Sentiment Analysis: Gauge the sentiment within the text data.
- Topic Modelling: Use techniques like LDA to uncover underlying topics.
- Predictive Modelling: Build models to classify or predict attributes based on the text data.

By applying this structured approach to text data, you can build a solid foundation for deeper analysis and advanced machine learning applications.

