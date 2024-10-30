import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')


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

def remove_non_ascii(text):
    # Remove non-ASCII characters using regex
    return re.sub(r'[^\x00-\x7F]', '', text)

def convert_to_lowercase(text):
    # Convert text to lowercase
    return text.lower()

def remove_noncontext_words(text):

    # Remove newline characters
    text = text.replace('\n', ' ')
    text = text.replace('&nbsp', ' ').replace('nbsp', ' ').replace('&lt', '').replace('&gt', '')
    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "")
    text = text.replace(",", "").replace("...", "").replace(":", "").replace(";", "").replace("!", ".").replace("?", ".")
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text_without_urls = re.sub(r"http\S+|\bhttps\S+", "", text) # Remove URLs

    # Tokenize the text into individual words
    words = text.split()

    # Define non-context words to remove
    noncontext_words = ['urllink', 'blog', 'date', 'maio']
    noncontext_words = noncontext_words + ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'septmber', 'october', 'november', 'december']
    noncontext_words = noncontext_words + ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    noncontext_words = noncontext_words + ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    noncontext_words = noncontext_words + ['nbsp', 'azon', 'acjc', 'alsob']
    
    # Remove noncontext words from the list of words
    filtered_words = [word for word in words if word not in noncontext_words]
  
    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
  
    return filtered_text

def remove_stop_words(text):
      
    # Tokenize the text into individual words
    words = text.split()

    # Stopwords + other words that don't add value to the analysis
    allnonwords = stopwords.words('english') + ['would', 'could', 'said', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    allnonwords = allnonwords + ['urllink', 'blog', 'date', 'me', 'you', 'she', 'her', 'him', 'they', 'them', 'their', 'your', 'yours', 'our', 'ours']
    allnonwords = allnonwords + ['right', 'look', 'first', 'last', 'never', 'thought', 'next', 'around', 'ever', 'always', 'come', 'to', 'thing', 'things', 'people']
    allnonwords = allnonwords + ['many', 'really', 'thing', 'much', 'stuff', 'there', 'hour', 'point', 'mine', 'yours', 'hers',
                                'his', 'theirs', 'ours', 'issue', 'thing', 'did', 'didnt', 'year', 'maio', 'thing', 'best', 'since', 'month', 'feel']
    allnonwords = allnonwords + ['much', 'something', 'someone', 'even', 'well', 'still', 'little', 'always', 'never', 'ever', 'sure', 'sort']
    allnonwords = allnonwords + ['every', 'anything', 'everything', 'nothing', 'everyone', 'everybody', 'everywhere', 'anyone', 'anybody']
    allnonwords = allnonwords + ['anywhere', 'someone', 'somebody', 'somewhere', 'nowhere', 'thing', 'something', 'nothing', 'everything']
    allnonwords = allnonwords + ['always', 'another', 'though', 'without', 'actually', 'do', 'dont', 'will', 'wont', 'can', 'cant']
    allnonwords = allnonwords + ['get', 'got', 'go', 'going', 'know', 'let', 'like', 'make', 'see', 'want', 'come', 'take', 'think']
    allnonwords = allnonwords + ['back', 'great', 'today', 'year', 'good', 'link', 'night', 'went', 'couple', 'say', 'give', 'need', 'make']
    allnonwords = allnonwords + ['youre', 'youve', 'youll', 'youd', 'hes', 'shes', 'its', 'were', 'theyre', 'thats', 'week', 'made', 'remember',
                                 'might', 'getting', 'better', 'real', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaahhhhhhhh', 'news', 'new', 'top',
                                 'u', 'day', 'brureau', 'love', 'u', 'do', 'not', 'well', 'fuck', 'na', 'haha', 'post', 'there',
                                 'anyway', 'ask', 'that', 'mean', 'dunno', 'file', 'miss', 'true', 'point', 'call', 'came', 'look',
                                 'site', 'na', 'talk', 'place', 'need', 'there', 'blog', 'entry', 'originally', 'posted', 'show', 'start',
                                 'okay', 'lots', 'finally', 'yippee', 'comes', 'hello', 'late', 'wish', 'weblog', 'damit', 'dammit',
                                 'currently', 'lala', 'opposite', 'told', 'update', 'updating', 'sometimes', 'maybe',
                                 'easy', 'half', 'different', 'called', 'total', 'took', 'word', 'done', 'stay', 'fine', 'find', 'cannot',
                                 'front', 'back', 'dude', 'feel', 'name', 'time', 'man', 'woman', 'home', 'ching', 'year',
                                 'times', 'yeah', 'sorry', 'whole', 'pretty', 'guess', 'nice', 'tomorrow', 'day']     


    # Remove stop words from the list of words
    filtered_words = [word for word in words if word not in allnonwords]
  
    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
  
    return filtered_text

def remove_short_words(text):
    # Split the text into words
    words = text.split()

    # Remove words with three letters or less
    filtered_words = [word for word in words if len(word) > 3]

    # Join the filtered words back into a text
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def remove_dates(text):
    # Define regex pattern to match months and days of the week
    pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'

    # Remove matched dates using regex substitution
    result = re.sub(pattern, '', text)
    return result

def remove_tags(text):
    # Remove HTML tags using regex
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'<*?>', '', text)
    return text

def remove_numbers(text):
    pattern = r'\d+'  # Regular expression pattern to match numbers
    result = re.sub(pattern, '', text)  # Replace the pattern with empty spaces
    return result

def remove_punctuation_and_newlines(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove newline characters
    # text = text.replace('\n', '')
    return text

def lemmatize_text(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each token in the text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the lemmatized tokens back into a single string
    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text

def stem_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Initialize Porter stemmer
    stemmer = PorterStemmer()

    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the stemmed words back into a text
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

# Functions to expand contractions e.g. don't -> do not
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# Remove very rare words (they are probably typos or noise)
def remove_less_frequent_words(text, num_words):
    # Count the occurrences of words in the text
    word_counts = Counter(text.split())

    # Collect the words to be removed
    words_to_remove = set()
    for word, count in word_counts.items():
        if count <= num_words:
            words_to_remove.add(word)

    # Remove the less frequent words from the text
    processed_text = ' '.join(word for word in text.split() if word not in words_to_remove)

    return processed_text


# Go through files and save all filenames in a list
def extract_filenames(directory):
    file_names = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.xml'):
            file_names.append(file_name)
    return file_names

# Iterate through the blog files in the directory and save demographic information as dataframe
def extract_demographics(directory, filenames):
    
    # Initialize demographic information
    blog_ids, genders, ages, educations, starsigns = [], [], [], [], []
    # Iterate through the blog files in the directory
    for filename in filenames:
         
        # Extract demographics from the filename and save to lists
        file_parts = filename.split('.')
        blog_id = file_parts[0]
        blog_ids.append(blog_id)
        gender = file_parts[1]
        genders.append(gender)
        age = int(file_parts[2])
        ages.append(age)
        education = file_parts[3]
        educations.append(education)
        starsign = file_parts[4]
        starsigns.append(starsign)

    data = {'ID': blog_ids, 'Gender': genders, 'Age': ages, 'Education': educations, 'Starsign': starsigns}
    return pd.DataFrame(data)

# Find nouns and lemmatize them, save to new list clean_content_nouns
def extract_nouns(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Perform part-of-speech tagging
    tagged_words = nltk.pos_tag(tokens)

    # Extract nouns
    nouns = [stemmer.stem(lemmatizer.lemmatize(word)) for word, pos in tagged_words if pos.startswith('NN')]

    return Counter(nouns)

# Find subjects of clauses and lemmatize them, and save to new list clean_content_subjects
def extract_subjects(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    subjects = []
    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)

        # Perform part-of-speech tagging
        tagged_words = nltk.pos_tag(words)

        # Extract subjects of clauses
        for i in range(len(tagged_words)):
            word, pos = tagged_words[i]
            if pos.startswith('V') and i > 0:
                prev_word, prev_pos = tagged_words[i - 1]
                if prev_pos.startswith('N') or prev_pos == 'PRP':
                    subject = stemmer.stem(lemmatizer.lemmatize(prev_word))
                    subjects.append(subject)

    return Counter(subjects)

def find_most_common_word_intext(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Tokenize, lemmatize and stem the text
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in word_tokenize(text)]

    # Count the occurrences of each word
    word_counts = Counter(words)
    return word_counts

# Find all clauses in each blog that have the identified topics in them
def find_clauses_with_word(text_list, word_list):
    if len(text_list) != len(word_list):
        raise ValueError("Number of texts and words should be the same.")

    result = []

    for text, word in zip(text_list, word_list):
        clauses = re.findall(r"[^.!?]+", text)  # Split the text into clauses
        matching_clauses = [clause.strip() for clause in clauses if word in clause]  # Find matching clauses
        matching_clauses = " .".join(matching_clauses)  # Join the clauses into a single string
        result.append([matching_clauses])

    return result

# Function to find most important words in list of texts using TF-IDF
def find_most_important_word_TFIDF(texts):
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts using the vectorizer
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Find the most important word in each text
    most_important_words = []
    for i in range(len(texts)):

        if len(texts[i]) > 0:

            # Get the TF-IDF scores for the current text
            tfidf_scores = tfidf_matrix[i].toarray()[0]

            # Find the index of the word with the highest TF-IDF score
            max_index = tfidf_scores.argmax()

            # Get the most important word
            most_important_word = feature_names[max_index]

        else:
            most_important_word = ''

        # Append the most important word to the list
        most_important_words.append(most_important_word)

    return most_important_words