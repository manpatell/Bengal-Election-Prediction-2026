from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_DATA_PATH = Path("data/raw/West_Bengal_AE.csv")
DEFAULT_BUNDLE_PATH = Path("models/bengal_election_bundle.pkl")

BOOLEAN_COLUMNS = ["Same_Constituency", "Same_Party", "Turncoat", "Incumbent", "Recontest"]
NUMERIC_COLUMNS = [
    "Year",
    "Assembly_No",
    "Constituency_No",
    "Age",
    "Votes",
    "Valid_Votes",
    "Electors",
    "N_Cand",
    "Turnout_Percentage",
    "Vote_Share_Percentage",
    "Margin_Percentage",
    "ENOP",
    "Contested",
    "No_Terms",
]
CATEGORICAL_COLUMNS = [
    "Party",
    "Constituency_Name",
    "Constituency_Type",
    "District_Name",
    "Sub_Region",
    "Sex",
    "Last_Party",
    "MyNeta_education",
    "TCPD_Prof_Main",
    "Election_Type",
]
PARTY_ALIASES = {
    "CPM": "CPIM",
    "FBL": "AIFB",
    "SUC": "SUCI",
    "AITC": "AITC",
}
EXCLUDED_PARTIES = {"NOTA", "NONE OF THE ABOVE"}


@dataclass
class CandidateConfig:
    name: str
    estimator: Any


def _normalize_party(value: Any) -> str:
    party = str(value or "").strip().upper()
    return PARTY_ALIASES.get(party, party)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text else "UNKNOWN"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().isin({"TRUE", "YES", "1"}).astype(int)


def load_election_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean_election_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _to_numeric(cleaned[column])

    for column in BOOLEAN_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _to_bool_int(cleaned[column])
        else:
            cleaned[column] = 0

    text_columns = {
        "Party",
        "Constituency_Name",
        "Constituency_Type",
        "District_Name",
        "Sub_Region",
        "Sex",
        "Last_Party",
        "MyNeta_education",
        "TCPD_Prof_Main",
        "Election_Type",
    }
    for column in text_columns:
        if column not in cleaned.columns:
            cleaned[column] = "UNKNOWN"
        cleaned[column] = cleaned[column].map(_normalize_text)

    cleaned["Party"] = cleaned["Party"].map(_normalize_party)
    cleaned["Last_Party"] = cleaned["Last_Party"].map(_normalize_party)
    cleaned["Won"] = (_to_numeric(cleaned.get("Position", 0)) == 1).astype(int)

    cleaned = cleaned[cleaned["Election_Type"].str.contains("ASSEMBLY", na=False)]
    cleaned = cleaned[~cleaned["Party"].isin(EXCLUDED_PARTIES)]
    cleaned = cleaned.dropna(subset=["Year", "Constituency_Name", "Party", "Vote_Share_Percentage"])
    cleaned = cleaned.sort_values(
        ["Constituency_Name", "Year", "Assembly_No", "Vote_Share_Percentage"],
        ascending=[True, True, True, False],
    )

    return cleaned.reset_index(drop=True)


def _build_previous_winner_map(df: pd.DataFrame) -> pd.DataFrame:
    winners = (
        df.loc[df["Won"] == 1, ["Constituency_Name", "Year", "Party", "Vote_Share_Percentage", "Margin_Percentage"]]
        .sort_values(["Constituency_Name", "Year"])
        .rename(
            columns={
                "Party": "Previous_Winner_Party_CurrentRow",
                "Vote_Share_Percentage": "Previous_Winner_Vote_Share_CurrentRow",
                "Margin_Percentage": "Previous_Winner_Margin_CurrentRow",
            }
        )
    )
    winners["Year"] = winners.groupby("Constituency_Name")["Year"].shift(-1)
    winners["Year"] = winners["Year"].astype("Int64")
    return winners.dropna(subset=["Year"])


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = clean_election_data(df)

    grouped_party = frame.groupby(["Constituency_Name", "Party"], sort=False)
    frame["Prev_Party_Vote_Share"] = grouped_party["Vote_Share_Percentage"].shift(1)
    frame["Prev_Party_Position"] = grouped_party["Position"].shift(1)
    frame["Prev_Party_Won"] = grouped_party["Won"].shift(1)
    frame["Prev_Party_Contests"] = grouped_party["Contested"].shift(1)
    frame["Prev_Party_No_Terms"] = grouped_party["No_Terms"].shift(1)

    grouped_constituency = frame.groupby("Constituency_Name", sort=False)
    frame["Prev_Constituency_Turnout"] = grouped_constituency["Turnout_Percentage"].shift(1)
    frame["Prev_Constituency_ENOP"] = grouped_constituency["ENOP"].shift(1)
    frame["Prev_Constituency_N_Cand"] = grouped_constituency["N_Cand"].shift(1)

    previous_winners = _build_previous_winner_map(frame)
    frame = frame.merge(previous_winners, on=["Constituency_Name", "Year"], how="left")
    frame["Is_Previous_Winner_Party"] = (
        frame["Party"] == frame["Previous_Winner_Party_CurrentRow"].fillna("UNKNOWN")
    ).astype(int)

    for column in [
        "Prev_Party_Vote_Share",
        "Prev_Party_Position",
        "Prev_Party_Won",
        "Prev_Party_Contests",
        "Prev_Party_No_Terms",
        "Prev_Constituency_Turnout",
        "Prev_Constituency_ENOP",
        "Prev_Constituency_N_Cand",
        "Previous_Winner_Vote_Share_CurrentRow",
        "Previous_Winner_Margin_CurrentRow",
    ]:
        frame[column] = _to_numeric(frame[column])

    frame["Vote_Share_Gap_vs_Previous"] = frame["Vote_Share_Percentage"] - frame["Prev_Party_Vote_Share"].fillna(
        frame["Vote_Share_Percentage"].median()
    )
    frame["Candidate_Experience_Index"] = frame["No_Terms"].fillna(0) + frame["Contested"].fillna(0)
    frame["Turnout_to_Elector_Ratio"] = np.where(
        frame["Electors"].fillna(0) > 0,
        frame["Valid_Votes"].fillna(0) / frame["Electors"].replace(0, np.nan),
        np.nan,
    )

    return frame


def _feature_columns() -> tuple[list[str], list[str]]:
    categorical = CATEGORICAL_COLUMNS + ["Previous_Winner_Party_CurrentRow"]
    numeric = NUMERIC_COLUMNS + [
        "Prev_Party_Vote_Share",
        "Prev_Party_Position",
        "Prev_Party_Won",
        "Prev_Party_Contests",
        "Prev_Party_No_Terms",
        "Prev_Constituency_Turnout",
        "Prev_Constituency_ENOP",
        "Prev_Constituency_N_Cand",
        "Previous_Winner_Vote_Share_CurrentRow",
        "Previous_Winner_Margin_CurrentRow",
        "Vote_Share_Gap_vs_Previous",
        "Candidate_Experience_Index",
        "Turnout_to_Elector_Ratio",
        "Is_Previous_Winner_Party",
    ] + BOOLEAN_COLUMNS
    return categorical, numeric


def _build_pipeline(config: CandidateConfig, categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", config.estimator)])


def _constituency_accuracy(df_eval: pd.DataFrame, probabilities: np.ndarray) -> float:
    scored = df_eval[["Constituency_Name", "Year", "Won"]].copy()
    scored["probability"] = probabilities
    best = (
        scored.sort_values(["Constituency_Name", "Year", "probability"], ascending=[True, True, False])
        .groupby(["Constituency_Name", "Year"], as_index=False)
        .first()
    )
    return float(best["Won"].mean()) if not best.empty else 0.0


def _safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 15) -> list[dict[str, float]]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(model, "feature_importances_"):
        return []

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)
    importance_df["importance"] = importance_df["importance"].astype(float)
    return importance_df.to_dict("records")


def train_and_save_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
) -> dict[str, Any]:
    raw_df = load_election_data(data_path)
    training_df = build_training_frame(raw_df)
    categorical_features, numeric_features = _feature_columns()
    features = categorical_features + numeric_features

    latest_year = int(training_df["Year"].max())
    validation_mask = training_df["Year"] == latest_year
    if validation_mask.sum() < 500:
        validation_mask = training_df["Year"] >= training_df["Year"].quantile(0.9)

    train_df = training_df.loc[~validation_mask].copy()
    validation_df = training_df.loc[validation_mask].copy()

    X_train = train_df[features]
    y_train = train_df["Won"].astype(int)
    X_validation = validation_df[features]
    y_validation = validation_df["Won"].astype(int)

    candidate_models = [
        CandidateConfig(
            name="random_forest_balanced",
            estimator=RandomForestClassifier(
                n_estimators=500,
                max_depth=16,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        ),
        CandidateConfig(
            name="random_forest_deep",
            estimator=RandomForestClassifier(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
        ),
        CandidateConfig(
            name="extra_trees_balanced",
            estimator=ExtraTreesClassifier(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]

    search_results: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_score = -1.0
    best_name = ""

    for candidate in candidate_models:
        pipeline = _build_pipeline(candidate, categorical_features, numeric_features)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        constituency_accuracy = _constituency_accuracy(validation_df, probabilities)
        row_f1 = float(f1_score(y_validation, probabilities >= 0.5, zero_division=0))
        search_results.append(
            {
                "model_name": candidate.name,
                "constituency_accuracy": constituency_accuracy,
                "row_f1": row_f1,
            }
        )
        score = constituency_accuracy + (row_f1 * 0.15)
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = candidate.name

    if best_pipeline is None:
        raise RuntimeError("Unable to fit any candidate model.")

    validation_probabilities = best_pipeline.predict_proba(X_validation)[:, 1]
    row_predictions = (validation_probabilities >= 0.5).astype(int)

    metrics = {
        "validation_year": int(validation_df["Year"].max()),
        "row_balanced_accuracy": float(balanced_accuracy_score(y_validation, row_predictions)),
        "row_precision": float(precision_score(y_validation, row_predictions, zero_division=0)),
        "row_recall": float(recall_score(y_validation, row_predictions, zero_division=0)),
        "row_f1": float(f1_score(y_validation, row_predictions, zero_division=0)),
        "constituency_accuracy": _constituency_accuracy(validation_df, validation_probabilities),
        "log_loss": float(log_loss(y_validation, validation_probabilities, labels=[0, 1])),
        "roc_auc": _safe_roc_auc(y_validation, validation_probabilities),
    }

    bundle = {
        "artifact_version": 2,
        "model_type": best_name,
        "model": best_pipeline,
        "feature_columns": features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "latest_year": latest_year,
        "metadata": {
            "metrics": metrics,
            "candidate_search": search_results,
            "feature_importance": _extract_feature_importance(best_pipeline),
            "training_rows": int(train_df.shape[0]),
            "validation_rows": int(validation_df.shape[0]),
            "all_rows": int(training_df.shape[0]),
            "available_years": sorted(training_df["Year"].dropna().astype(int).unique().tolist()),
        },
    }

    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    return bundle


def load_bundle(bundle_path: str | Path = DEFAULT_BUNDLE_PATH) -> dict[str, Any] | None:
    path = Path(bundle_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_legacy_bundle(
    model_path: str | Path = "models/bengal_rf_model.pkl",
    encoder_path: str | Path = "models/encoders.pkl",
) -> dict[str, Any] | None:
    model_path = Path(model_path)
    encoder_path = Path(encoder_path)
    if not model_path.exists() or not encoder_path.exists():
        return None

    with model_path.open("rb") as model_handle:
        model = pickle.load(model_handle)
    with encoder_path.open("rb") as encoder_handle:
        encoders = pickle.load(encoder_handle)

    return {
        "artifact_version": 1,
        "model_type": "legacy_random_forest",
        "model": model,
        "encoders": encoders,
        "metadata": {
            "warning": "Using the legacy artifact. Run train_model.py to enable the full reliability layer.",
        },
    }


def build_scenario_frame(
    prepared_df: pd.DataFrame,
    constituency_name: str,
    swing_adjustments: dict[str, float] | None = None,
    tracked_parties: list[str] | None = None,
) -> pd.DataFrame:
    latest_year = int(prepared_df["Year"].max())
    constituency_rows = prepared_df[
        (prepared_df["Constituency_Name"] == _normalize_text(constituency_name))
        & (prepared_df["Year"] == latest_year)
        & (~prepared_df["Party"].isin(EXCLUDED_PARTIES))
    ].copy()
    constituency_rows = constituency_rows.sort_values("Vote_Share_Percentage", ascending=False)

    if constituency_rows.empty:
        raise ValueError(f"No latest election rows available for constituency '{constituency_name}'.")

    if tracked_parties is None:
        tracked_parties = constituency_rows["Party"].head(5).tolist()

    swing_adjustments = {(_normalize_party(party)): float(value) for party, value in (swing_adjustments or {}).items()}
    constituency_rows["Base_Vote_Share"] = constituency_rows["Vote_Share_Percentage"]

    for party, adjustment in swing_adjustments.items():
        mask = constituency_rows["Party"] == party
        constituency_rows.loc[mask, "Vote_Share_Percentage"] = (
            constituency_rows.loc[mask, "Vote_Share_Percentage"] + adjustment
        ).clip(lower=0.0)

    total_before = constituency_rows["Base_Vote_Share"].sum()
    total_after = constituency_rows["Vote_Share_Percentage"].sum()
    if total_before > 0 and total_after > 0:
        constituency_rows["Vote_Share_Percentage"] = constituency_rows["Vote_Share_Percentage"] * (
            total_before / total_after
        )

    constituency_rows["Tracked"] = constituency_rows["Party"].isin([_normalize_party(party) for party in tracked_parties])
    constituency_rows["Vote_Share_Gap_vs_Previous"] = constituency_rows["Vote_Share_Percentage"] - constituency_rows[
        "Prev_Party_Vote_Share"
    ].fillna(constituency_rows["Vote_Share_Percentage"].median())
    constituency_rows["Is_Previous_Winner_Party"] = (
        constituency_rows["Party"] == constituency_rows["Previous_Winner_Party_CurrentRow"].fillna("UNKNOWN")
    ).astype(int)
    return constituency_rows


def predict_scenario(bundle: dict[str, Any], scenario_df: pd.DataFrame) -> pd.DataFrame:
    if bundle.get("artifact_version") == 1:
        encoders = bundle["encoders"]
        supported = scenario_df[
            scenario_df["Party"].isin(encoders["party"].classes_)
            & scenario_df["Constituency_Name"].isin(encoders["constituency"].classes_)
        ].copy()
        supported["Constituency_Code"] = encoders["constituency"].transform(supported["Constituency_Name"])
        supported["Party_Code"] = encoders["party"].transform(supported["Party"])
        prediction_frame = supported[["Constituency_Code", "Party_Code", "Vote_Share_Percentage", "Age"]].fillna(0)
        supported["Win_Probability"] = bundle["model"].predict_proba(prediction_frame)[:, 1]
        return supported.sort_values("Win_Probability", ascending=False)

    features = bundle["feature_columns"]
    scenario_features = scenario_df.reindex(columns=features)
    scenario_df = scenario_df.copy()
    scenario_df["Win_Probability"] = bundle["model"].predict_proba(scenario_features)[:, 1]
    return scenario_df.sort_values("Win_Probability", ascending=False)


def evaluate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    return bundle.get("metadata", {})
