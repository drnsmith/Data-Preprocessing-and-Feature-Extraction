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
