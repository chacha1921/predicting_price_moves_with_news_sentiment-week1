# 📊 Predicting Price Movements with News Sentiment — Week 1

## 📌 Objective

This project integrates financial news sentiment with historical stock price data to analyze market behavior and explore predictive insights. The workflow includes technical indicators, sentiment analysis of news headlines, and correlation studies to understand stock price movements.

---

## ⚙️ Setup Instructions

```bash
git clone <your-repo-url>
cd predicting_price_moves_with_news_sentiment-week1

# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


├── data/                     # Raw & processed datasets
├── notebooks/                # Jupyter Notebooks for EDA & Analysis
├── src/                      # Reusable Python modules
│   ├── finance_utils.py      # Financial indicators & visualizations
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
