# tests/test_eda.py

import pandas as pd
from datetime import datetime

# Sample dummy data for testing
sample_data = {
    'headline': ['Apple hits new high', 'Tesla drops after earnings', 'FDA approves new drug'],
    'url': ['http://a.com', 'http://b.com', 'http://c.com'],
    'publisher': ['reuters', 'cnn', 'healthline'],
    'date': ['2024-05-01 12:00:00-04:00', '2024-05-01 14:00:00-04:00', '2024-05-02 09:00:00-04:00'],
    'stock': ['AAPL', 'TSLA', 'PFE']
}

df = pd.DataFrame(sample_data)

def test_no_missing_values():
    assert df.isnull().sum().sum() == 0, "There should be no missing values in the sample data."

def test_headline_length_computation():
    df['headline_length'] = df['headline'].apply(len)
    assert 'headline_length' in df.columns, "headline_length column not created"
    assert df['headline_length'].iloc[0] == len('Apple hits new high'), "Incorrect headline length calculated"

def test_datetime_conversion():
    df['date'] = pd.to_datetime(df['date'], utc=True)
    assert pd.api.types.is_datetime64_any_dtype(df['date']), "date column is not datetime after conversion"
    assert df['date'].dt.tz is not None, "Timezone info missing after datetime conversion"

def test_hour_extraction():
    df['hour'] = df['date'].dt.hour
    assert 'hour' in df.columns, "hour column not created"
    assert df['hour'].iloc[0] == 16, "Incorrect hour extracted from date"  # 12 PM UTC-4 = 16 UTC

def test_tokenization():
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

    df['tokens'] = df['headline'].apply(clean_text)
    assert isinstance(df['tokens'].iloc[0], list), "Tokenization did not return a list"
    assert 'apple' in df['tokens'].iloc[0], "Token 'apple' not found after tokenization"

