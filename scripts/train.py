"""Entry point for training the fantasy points prediction model."""

from nfldata.features import build_features
from nfldata.model import train_model

# CR opus: Hardcoded range stops at 2025 (exclusive), so it only uses 2018-2024 data.
# CR opus: The 2025 season is now complete; this should be range(2018, 2026).
df = build_features(range(2018, 2025))
model, importance = train_model(df)

print("\n=== Top 20 Most Important Features ===")
print(importance.head(20))
