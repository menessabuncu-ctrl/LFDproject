"""
Amazon Product Reviews Scraper
Learning from Data - Final Project
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json


class AmazonReviewScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.reviews = []

    def scrape_reviews_beautifulsoup(self, product_urls, max_reviews=3000):
        """
        Scrape reviews using BeautifulSoup
        Note: This is a template - actual Amazon scraping requires handling CAPTCHAs
        """
        print("Starting review scraping...")

        for url in product_urls:
            try:
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find review elements (selectors may need updates)
                review_divs = soup.find_all('div', {'data-hook': 'review'})

                for review in review_divs:
                    try:
                        rating = review.find('i', {'data-hook': 'review-star-rating'})
                        rating_text = rating.text.strip() if rating else None

                        title = review.find('a', {'data-hook': 'review-title'})
                        title_text = title.text.strip() if title else ""

                        body = review.find('span', {'data-hook': 'review-body'})
                        body_text = body.text.strip() if body else ""

                        if rating_text and body_text:
                            self.reviews.append({
                                'rating': float(rating_text.split()[0]),
                                'title': title_text,
                                'text': body_text,
                                'full_text': f"{title_text} {body_text}"
                            })
                    except Exception as e:
                        continue

                time.sleep(random.uniform(2, 4))  # Rate limiting

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue

        return pd.DataFrame(self.reviews)

    def scrape_reviews_selenium(self, product_urls, max_reviews=3000):
        """
        Alternative: Scrape using Selenium for dynamic content
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)

        for url in product_urls:
            try:
                driver.get(url)
                time.sleep(3)

                # Scroll to load more reviews
                for _ in range(5):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)

                review_elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')

                for review in review_elements:
                    try:
                        rating = review.find_element(By.CSS_SELECTOR, '[data-hook="review-star-rating"]').text
                        text = review.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]').text

                        self.reviews.append({
                            'rating': float(rating.split()[0]),
                            'text': text
                        })
                    except:
                        continue

            except Exception as e:
                print(f"Error: {e}")
                continue

        driver.quit()
        return pd.DataFrame(self.reviews)

    def create_sentiment_labels(self, df):
        """
        Convert ratings to sentiment labels
        1-2 stars: Negative (0)
        3 stars: Neutral (1)
        4-5 stars: Positive (2)
        """

        def rating_to_sentiment(rating):
            if rating <= 2:
                return 0  # Negative
            elif rating == 3:
                return 1  # Neutral
            else:
                return 2  # Positive

        df['sentiment'] = df['rating'].apply(rating_to_sentiment)
        df['sentiment_label'] = df['sentiment'].map({0: 'negative', 1: 'neutral', 2: 'positive'})
        return df

    def save_data(self, df, filename='amazon_reviews.csv'):
        """Save scraped data"""
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} reviews to {filename}")
        return df


# Alternative: Load pre-collected dataset
def load_alternative_dataset():
    """
    For academic purposes, you can use publicly available datasets:
    - Amazon Review Dataset (Kaggle)
    - Amazon Product Data (Stanford)
    """
    # Example: Loading from Kaggle dataset
    try:
        df = pd.read_csv('amazon_reviews_dataset.csv')
        print(f"Loaded {len(df)} reviews from existing dataset")
        return df
    except:
        print("Please download dataset from Kaggle or collect your own data")
        return None


# Simulated data for demonstration
def create_sample_data(n_samples=3000):
    """
    Create sample data for testing (replace with real scraping)
    """
    import numpy as np

    positive_samples = [
                           "This product is absolutely amazing! Highly recommended.",
                           "Excellent quality, exceeded my expectations.",
                           "Best purchase ever! Love it!",
                           "Outstanding product, will buy again.",
                           "Perfect! Exactly what I needed.",
                       ] * 400

    negative_samples = [
                           "Terrible quality, waste of money.",
                           "Very disappointed, does not work as advertised.",
                           "Poor quality, broke after one use.",
                           "Not worth the price, very bad.",
                           "Horrible experience, would not recommend.",
                       ] * 300

    neutral_samples = [
                          "It's okay, nothing special.",
                          "Average product, does the job.",
                          "Not bad, but not great either.",
                          "Decent for the price.",
                          "Works as expected, nothing more.",
                      ] * 200

    texts = positive_samples + negative_samples + neutral_samples
    ratings = [5] * len(positive_samples) + [1] * len(negative_samples) + [3] * len(neutral_samples)

    df = pd.DataFrame({
        'text': texts[:n_samples],
        'rating': ratings[:n_samples]
    })

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Option 1: Scrape real data (requires handling Amazon's anti-scraping measures)
    scraper = AmazonReviewScraper()

    # Option 2: Use existing dataset
    # df = load_alternative_dataset()

    # Option 3: Create sample data for demonstration
    df = create_sample_data(3000)

    # Add sentiment labels
    scraper_instance = AmazonReviewScraper()
    df = scraper_instance.create_sentiment_labels(df)

    # Save data
    df.to_csv('data/raw_reviews.csv', index=False)
    print(f"Dataset created with {len(df)} samples")
    print(df['sentiment_label'].value_counts())