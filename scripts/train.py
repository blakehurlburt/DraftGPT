"""Entry point for training the fantasy points prediction model."""

from nfldata.features import build_features
from nfldata.model import train_model

df = build_features(range(2018, 2026))
model, importance = train_model(df)

print("\n=== Top 20 Most Important Features ===")
print(importance.head(20))
