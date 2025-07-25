# Fake News Classifier
This project focuses on building a machine learning model to classify news articles as either "Fake News" or "Factual News."
The goal is to leverage Natural Language Processing (NLP) techniques and various classification algorithms to accurately distinguish between the two categories.

This project was developed as part of the 365 Data Science AI Engineer Bootcamp Course.

# Project Overview
In an era of pervasive digital information, the ability to discern factual news from misinformation is crucial.
This project addresses this challenge by developing a robust fake news classifier.
It involves a comprehensive approach, starting from data loading and exploration, through advanced NLP techniques like Part-of-Speech (POS) tagging, Named Entity Recognition (NER), 
Sentiment Analysis, and Topic Modeling, to Feature Engineering and finally, Training and Evaluating several Machine Learning Models.

# Dataset
The project utilizes a dataset named ```fake_news_data.csv```, which contains news articles categorized as "Fake News" or "Factual News".
The dataset includes columns such as ```title```, ```text```, ```date```, and ```fake_or_factual```.

```data.head()```: Displays the first few rows of the dataset, showing the structure and content.

```data.info()```: Provides a summary of the DataFrame, including data types and non-null counts, confirming 198 entries with no missing values.

```data['fake_or_factual'].value_counts()```: Shows the distribution of "Fake News" and "Factual News" entries, indicating a balanced dataset.

# Methodology
## Data Loading and Initial Exploration
The project begins by loading the ```fake_news_data.csv``` file into a Pandas DataFrame.
Initial checks are performed to understand the data's structure, identify missing values, and examine the distribution of the target variable (```fake_or_factual```).

## Data Exploration (POS and NER)
Advanced NLP techniques are applied using **spaCy** for *Part-of-Speech (POS) Tagging* and *Named Entity Recognition (NER) Tagging*.
This step aims to uncover linguistic patterns and entity distributions unique to fake and factual news.

- POS Tagging:

   - News articles are separated into ```fake_news``` and ```fact_news``` DataFrames.

   - *spaCy* pipeline (```nlp.pipe```) is used to process the text and extract tokens, NER tags, and POS tags.

   - Frequency counts of tokens and POS tags are analyzed for both fake and factual news to identify common linguistic characteristics.

   - **Observations**: While general POS tag frequencies (NOUN, VERB, PROPN) are similar, the specific words associated with these tags often differ in their rankings between fake and factual news. For instance, '*government*' appears in different frequency positions.
- Named Entities:

   - Top named entities (e.g., ORG, GPE, PERSON) are extracted and analyzed for both datasets.

   - An ```ner_palette``` is defined for visualization purposes.
     
## Text Preprocessing and Cleaning
A crucial step in NLP, this phase involves:

- **Lowercasing**: Converting all text to lowercase to ensure uniformity.
- **Removing URLs**: Eliminating web links from the text.
- **Removing HTML Tags**: Stripping any HTML markup.
- **Removing Punctuations**: Deleting punctuation marks.
- **Removing Stopwords**: Removing common words (like "the", "is", "a") that do not carry significant meaning.
- **Lemmatization**: Reducing words to their base or root form (e.g., "running" to "run") using ```WordNetLemmatizer```.
- **Tokenization**: Breaking down text into individual words or tokens using ```nltk.word_tokenize```.


## Sentiment Analysis
The VADER (Valence Aware Dictionary and Sentiment Reasoner) sentiment analysis tool is used to analyze the emotional tone of the news articles.
- **VADER Scores**: For each article, negative, neutral, and positive sentiment scores are calculated.
- **Analysis**: The distribution of sentiment scores is visualized to identify potential differences in emotional content.

## Topic Modeling (LSA)
Latent Semantic Analysis (LSA) is used to identify underlying topics within the news articles.
- **TF-IDF Vectorization**: Text data is transformed into TF-IDF (Term Frequency-Inverse Document Frequency) vectors, which reflect the importance of words in a document relative to the entire corpus.
- **LSI Model**: An LSI (Latent Semantic Indexing) model is trained to reduce dimensionality and extract latent topics.
- **Coherence Score**: The coherence of the topics is evaluated to determine the quality of the topic model.
  
## Feature Engineering
Beyond raw text, additional features are engineered to enhance model performance:
- **Word Count**: Number of words in each article.
- **Character Count**: Number of characters in each article.
- **Average Word Length**: Average length of words in each article.
- **Sentiment Scores**: VADER negative, neutral, and positive scores.
- **TF-IDF Features**: Features derived from the TF-IDF vectorization.
- **Count Vectorizer Features**: Features derived from Count Vectorizer.

These features are combined to create a comprehensive feature set for classification.

## Model Training and Evaluation
The dataset is split into **training** and **testing** sets.
Several machine learning models are trained and evaluated based on their *accuracy* and *classification reports*.

## Models Used
The following classification models were implemented and evaluated:

1. Logistic Regression: A linear model used for binary classification.

     - TF-IDF Vectorizer: Used for text feature extraction.

     - Results: Achieved an **accuracy of 0.83**, with **precision**, **recall**, and **F1-scores** around **0.83-0.84** for both classes.

2. Stochastic Gradient Descent (SGD) Classifier: An efficient linear classifier that supports various loss functions and penalties.

     - TF-IDF Vectorizer: Used for text feature extraction.

     - Results: Achieved an **accuracy of 0.83**, with **precision**, **recall**, and **F1-scores** around **0.83-0.84** for both classes.

3. Support Vector Machine (SVM): A powerful and versatile machine learning model for classification.

    - TF-IDF Vectorizer: Used for text feature extraction.

    - Results: Achieved an **accuracy of 0.83**, with **precision**, **recall**, and **F1-scores** around **0.82-0.85** for both classes.

## Results
All three models (Logistic Regression, SGD Classifier, and SVM) performed similarly, achieving an **accuracy** of approximately **83%** on the test set.
The classification reports indicate good balance between **precision** and **recall** for both "Factual News" and "Fake News" classes.

## Dependencies
The project requires the following Python libraries:
- ```pandas```
- ```matplotlib```
- ```seaborn```
- ```spacy``` (with ```en_core_web_sm``` model)
- ```re```
- ```nltk``` (with ```punkt```, ```wordnet```, ```stopwords``` downloads)
- ```vaderSentiment```
- ```gensim```
- ```sklearn```
  
You can install these dependencies using ```pip```:
```
pip install pandas matplotlib seaborn spacy nltk vaderSentiment gensim scikit-learn
python -m spacy download en_core_web_sm
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```
## Usage
1. **Clone the repositiory**:
   ```
   git clone https://github.com/debb-major/nlp-fake-news-classifier.git
   cd nlp-fake-news-classifier
   ```
2. **Place the dataset**: Ensure ```fake_news_data.csv``` is in the ```dataset/``` directory relative to your notebook.

3. **Install dependencies**: Follow the instructions in the ```Dependencies``` section.

4. **Run the Jupyter Notebook**: Open and run the ```fake_or_fact_news.ipynb``` notebook cell by cell to reproduce the analysis and model training.

## Acknowledgments
This project was completed as part of the **365 Data Science AI Engineer Bootcamp Course**.
Special thanks to the instructors and the curriculum for providing the foundational knowledge and guidance necessary to undertake this project.
