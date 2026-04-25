from typing import Dict

import pandas as pd


EXPECTED_FEATURE_COLUMNS = [f"q{i+1}" for i in range(18)]

QUESTION_LABELS = {
    "q1": "Nervous/Anxious",
    "q2": "Sad/Depressed",
    "q3": "Irritable/Angry",
    "q4": "Headaches/Body pain",
    "q5": "Physical fatigue",
    "q6": "Sleep trouble",
    "q7": "Rapid heartbeat",
    "q8": "Difficulty concentrating",
    "q9": "Negative thoughts",
    "q10": "Worry about future",
    "q11": "Difficulty with decisions",
    "q12": "Appetite changes",
    "q13": "Avoiding social",
    "q14": "Overwhelmed by tasks",
    "q15": "Work-life balance",
    "q16": "Work/study stress",
    "q17": "Relationship stress",
    "q18": "Financial stress",
}

QUESTION_CATEGORIES = {
    "emotional": ["q1", "q2", "q3"],
    "physical": ["q4", "q5", "q6", "q7"],
    "cognitive": ["q8", "q9", "q10", "q11"],
    "behavioral": ["q12", "q13", "q14"],
    "stressors": ["q15", "q16", "q17", "q18"],
}

# Relative importance for weighted questionnaire scoring and for the
# questionnaire model's input preprocessing.
_RAW_QUESTION_WEIGHTS = {
    "q1": 1.05,
    "q2": 1.10,
    "q3": 0.95,
    "q4": 0.85,
    "q5": 0.95,
    "q6": 1.20,
    "q7": 1.10,
    "q8": 1.00,
    "q9": 1.20,
    "q10": 1.05,
    "q11": 0.95,
    "q12": 0.85,
    "q13": 0.95,
    "q14": 1.20,
    "q15": 0.95,
    "q16": 1.15,
    "q17": 1.00,
    "q18": 1.00,
}

_WEIGHT_NORMALIZER = len(_RAW_QUESTION_WEIGHTS) / sum(_RAW_QUESTION_WEIGHTS.values())
QUESTION_WEIGHTS: Dict[str, float] = {
    question: weight * _WEIGHT_NORMALIZER
    for question, weight in _RAW_QUESTION_WEIGHTS.items()
}


def apply_question_weights(features: pd.DataFrame) -> pd.DataFrame:
    """Scale questionnaire features using the shared question weights."""
    missing_columns = [
        column for column in EXPECTED_FEATURE_COLUMNS if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing questionnaire feature columns: {missing_columns}")

    weighted = features.loc[:, EXPECTED_FEATURE_COLUMNS].copy()
    for column in EXPECTED_FEATURE_COLUMNS:
        weighted[column] = weighted[column].astype(float) * QUESTION_WEIGHTS[column]
    return weighted
