"""
Data Preprocessing and Feature Engineering
Learning from Data - Final Project
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens):
        """Apply lemmatization"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text, remove_stops=True, use_lemma=True):
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)

        # Lemmatize or stem
        if use_lemma:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)

        return ' '.join(tokens)


class FeatureExtractor:
    def __init__(self):
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None

    def extract_bow_features(self, texts, max_features=5000, fit=True):
        """Extract Bag-of-Words features"""
        if fit or self.bow_vectorizer is None:
            self.bow_vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2
            )
            features = self.bow_vectorizer.fit_transform(texts)
        else:
            features = self.bow_vectorizer.transform(texts)

        return features

    def extract_tfidf_features(self, texts, max_features=5000, fit=True):
        """Extract TF-IDF features"""
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)

        return features

    def extract_custom_features(self, texts):
        """Extract custom domain-specific features"""
        features = []

        for text in texts:
            feature_dict = {
                'length': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
                'contains_not': 1 if 'not' in text.lower() else 0,
                'contains_excellent': 1 if 'excellent' in text.lower() else 0,
                'contains_terrible': 1 if 'terrible' in text.lower() else 0,
                'contains_love': 1 if 'love' in text.lower() else 0,
                'contains_hate': 1 if 'hate' in text.lower() else 0,
            }
            features.append(list(feature_dict.values()))

        return np.array(features)


class DataAnalyzer:
    @staticmethod
    def analyze_dataset(df):
        """Analyze dataset statistics"""
        print("=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"\nClass distribution:")
        print(df['sentiment_label'].value_counts())
        print(f"\nClass distribution (%):")
        print(df['sentiment_label'].value_counts(normalize=True) * 100)

        print(f"\nText length statistics:")
        df['text_length'] = df['text'].apply(len)
        print(df['text_length'].describe())

        print(f"\nWord count statistics:")
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        print(df['word_count'].describe())

    @staticmethod
    def visualize_data(df):
        """Create visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Class distribution
        df['sentiment_label'].value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')

        # Text length distribution by sentiment
        for sentiment in df['sentiment_label'].unique():
            data = df[df['sentiment_label'] == sentiment]['text_length']
            axes[0, 1].hist(data, alpha=0.5, label=sentiment, bins=30)
        axes[0, 1].set_title('Text Length Distribution by Sentiment')
        axes[0, 1].set_xlabel('Text Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Word count distribution
        df['word_count'].hist(bins=50, ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('Word Count Distribution')
        axes[1, 0].set_xlabel('Word Count')
        axes[1, 0].set_ylabel('Frequency')

        # Rating distribution
        if 'rating' in df.columns:
            df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='lightgreen')
            axes[1, 1].set_title('Rating Distribution')
            axes[1, 1].set_xlabel('Rating')
            axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig('results/data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def prepare_data(csv_path='data/raw_reviews.csv', test_size=0.2, val_size=0.15):
    """
    Complete data preparation pipeline
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Analyze dataset
    analyzer = DataAnalyzer()
    analyzer.analyze_dataset(df)
    analyzer.visualize_data(df)

    # Preprocess text
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()
    df['clean_text'] = df['text'].apply(lambda x: preprocessor.preprocess(x))

    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()

    # Split data first
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['clean_text'], df['sentiment'],
        test_size=test_size, random_state=42, stratify=df['sentiment']
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size), random_state=42, stratify=y_temp
    )

    print(f"\nData split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    # Extract different features
    features = {
        'bow': {
            'X_train': feature_extractor.extract_bow_features(X_train, fit=True),
            'X_val': feature_extractor.extract_bow_features(X_val, fit=False),
            'X_test': feature_extractor.extract_bow_features(X_test, fit=False)
        },
        'tfidf': {
            'X_train': feature_extractor.extract_tfidf_features(X_train, fit=True),
            'X_val': feature_extractor.extract_tfidf_features(X_val, fit=False),
            'X_test': feature_extractor.extract_tfidf_features(X_test, fit=False)
        }
    }

    # Save processed data
    data = {
        'features': features,
        'labels': {
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values
        },
        'raw_text': {
            'X_train': X_train.values,
            'X_val': X_val.values,
            'X_test': X_test.values
        },
        'preprocessor': preprocessor,
        'feature_extractor': feature_extractor
    }

    return data


if __name__ == "__main__":
    # Prepare data
    data = prepare_data('data/raw_reviews.csv')

    # Save preprocessed data
    import pickle

    with open('data/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("\nData preprocessing completed!")
    print(f"Features saved: BoW and TF-IDF")