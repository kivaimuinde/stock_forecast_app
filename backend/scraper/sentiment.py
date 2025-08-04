# backend/scraper/sentiment.py

from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Basic Sentiment Analysis ----------
def analyze_sentiment(text):
    """
    Returns polarity score (-1 to 1) for a given text.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# ---------- NewsAPI Integration ----------
from newsapi import NewsApiClient

def get_newsapi_sentiment(query="stocks"):
    """
    Fetch news using NewsAPI and calculate average sentiment.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return 0  # fallback if key not set

    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)

    headlines = [article['title'] for article in articles['articles']]
    if not headlines:
        return 0  # neutral if nothing fetched

    sentiments = [analyze_sentiment(title) for title in headlines]
    return sum(sentiments) / len(sentiments)

# ---------- Twitter API Integration ----------
import tweepy

def get_twitter_sentiment(query="#stocks"):
    """
    Fetch tweets using Twitter API and calculate average sentiment.
    """
    bearer_token = os.getenv("TWITTER_BEARER")
    if not bearer_token:
        return 0

    client = tweepy.Client(bearer_token=bearer_token)
    tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["text"])

    if not tweets.data:
        return 0

    sentiments = [analyze_sentiment(tweet.text) for tweet in tweets.data]
    return sum(sentiments) / len(sentiments)
