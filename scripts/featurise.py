"""DVC stage: load raw voters, run ColumnTransformer, save processed features + pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import joblib
import pandas as pd

from src.features.pipeline import build_feature_pipeline, engineer_features, FEATURE_COLS, TARGET_COL

RAW_PATH = Path("data/raw/voters.parquet")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main() -> None:
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_parquet(RAW_PATH)
    print(f"  {len(df):,} rows loaded")

    df_eng = engineer_features(df.copy())
    X = df_eng[FEATURE_COLS]
    y = df_eng[TARGET_COL]

    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)
    print(f"  Features transformed: {X_transformed.shape}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_df = pd.DataFrame(X_transformed)
    features_df["target"] = y.values
    out_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"  Saved features -> {out_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipe_path = MODELS_DIR / "pipeline.pkl"
    joblib.dump(pipeline, pipe_path)
    print(f"  Saved pipeline -> {pipe_path}")

    print("Featurisation complete.")


if __name__ == "__main__":
    main()
