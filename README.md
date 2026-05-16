# Sentiment Analysis with RoBERTa

Sentiment analysis pipeline using the **cardiffnlp/twitter-roberta-base-sentiment-latest** model from Hugging Face — a RoBERTa model fine-tuned on ~124M tweets.

## Labels

- **Positive** — upbeat, favourable sentiment
- **Neutral** — factual or mixed
- **Negative** — critical or unfavourable sentiment

## Quick Start

```bash
pip install transformers torch scipy
```

```python
from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

results = sentiment([
    "I love this product!",
    "The weather is okay today.",
    "Terrible experience, would not recommend."
])

for r in results:
    print(r['label'], f"{r['score']:.2%}")
# POSITIVE  98.7%
# NEUTRAL   71.2%
# NEGATIVE  96.1%
```

## Batch Processing

```python
texts = ["Great!", "Not bad", "Awful"]
results = sentiment(texts, batch_size=8, truncation=True, max_length=512)
```

## Requirements

```
transformers>=4.30.0
torch>=2.0.0
scipy
```

## Model Card

[cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) on Hugging Face.
