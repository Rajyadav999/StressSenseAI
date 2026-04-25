import os
import json
import importlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from .model_artifacts import (
    current_sklearn_version,
    legacy_stress_meta_path,
    legacy_stress_model_path,
    legacy_stress_shap_model_path,
    shared_stress_meta_path,
    shared_stress_model_path,
    shared_stress_shap_model_path,
    stress_meta_path,
    stress_model_path,
    stress_shap_model_path,
)
from .stress_forecaster import stress_forecaster
from .questionnaire_config import (
    EXPECTED_FEATURE_COLUMNS,
    QUESTION_CATEGORIES,
    QUESTION_LABELS,
    QUESTION_WEIGHTS,
    apply_question_weights,
)
from .sklearn_pickle import load_sklearn_pickle, sha256_file


class StressPredictor:
    def __init__(self):
        self.model = None
        self.shap_model = None  # tree-based model for SHAP
        self.shap_explainer = None
        self.model_path = stress_model_path()
        self.shap_model_path = stress_shap_model_path()
        self.meta_path = stress_meta_path()
        self.shared_model_path = shared_stress_model_path()
        self.shared_shap_model_path = shared_stress_shap_model_path()
        self.shared_meta_path = shared_stress_meta_path()
        self.legacy_model_path = legacy_stress_model_path()
        self.legacy_shap_model_path = legacy_stress_shap_model_path()
        self.legacy_meta_path = legacy_stress_meta_path()
        self.sklearn_version = current_sklearn_version()
        self.stress_labels = {
            0: "Low",
            1: "Moderate",
            2: "High",
            3: "Severe",
        }
        self.load_model()

    def _extract_tree_model_from_primary(self):
        """Reuse the fitted RF already embedded in the primary ensemble when available."""
        model = self.model
        if model is None:
            return None

        base_model = None
        if hasattr(model, "named_estimators_"):
            base_model = model
        elif hasattr(model, "estimator"):
            estimator = model.estimator
            base_model = getattr(estimator, "estimator", estimator)

        if base_model is None or not hasattr(base_model, "named_estimators_"):
            return None

        rf_model = base_model.named_estimators_.get("rf")
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            return rf_model
        return None

    @staticmethod
    def _sha256_file(path: str) -> str:
        return sha256_file(path)

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as meta_file:
                return json.load(meta_file)
        except Exception:
            return {}

    def _expected_hash(self, meta: Dict[str, Any], key: str) -> str:
        meta_hash = str(meta.get(key, "") or "").strip()
        env_key = {
            "model_sha256": "STRESS_MODEL_SHA256",
            "shap_model_sha256": "STRESS_SHAP_MODEL_SHA256",
        }.get(key, "")
        env_hash = os.getenv(env_key, "").strip() if env_key else ""
        return env_hash or meta_hash

    def _is_runtime_compatible(self, meta: Dict[str, Any]) -> bool:
        artifact_version = str(meta.get("sklearn_version", "") or "").strip()
        if not artifact_version:
            return True
        return artifact_version == self.sklearn_version

    def _model_candidates(self) -> List[Tuple[Path, Path]]:
        return [
            (self.model_path, self.meta_path),
            (self.shared_model_path, self.shared_meta_path),
            (self.legacy_model_path, self.legacy_meta_path),
        ]

    def _shap_model_candidates(self) -> List[Tuple[Path, Path]]:
        return [
            (self.shap_model_path, self.meta_path),
            (self.shared_shap_model_path, self.shared_meta_path),
            (self.legacy_shap_model_path, self.legacy_meta_path),
        ]

    def _load_pickle_with_integrity(self, path: str, expected_hash: str = ""):
        return load_sklearn_pickle(path, expected_hash=expected_hash)

    def load_model(self):
        """Load the trained model, retraining automatically if the pickle is invalid."""
        for candidate, meta_path in self._model_candidates():
            if not candidate.exists():
                continue
            meta = self._load_metadata(meta_path)
            if not self._is_runtime_compatible(meta):
                print(
                    "Skipping ML model artifact from a different scikit-learn runtime: "
                    f"{candidate} (artifact={meta.get('sklearn_version')}, runtime={self.sklearn_version})"
                )
                continue
            try:
                expected = self._expected_hash(meta, "model_sha256")
                self.model = self._load_pickle_with_integrity(str(candidate), expected)
                self.model_path = candidate
                self.meta_path = meta_path
                print(f"ML model loaded successfully from {candidate}")
                break
            except Exception as exc:
                print(f"Failed to load ML model from {candidate}: {exc}")
        else:
            print("No compatible ML model found. Training a new model.")
            self._retrain_model()

        # Reuse the RF already inside the ensemble to avoid loading a second
        # large tree artifact on the hot path.
        self.shap_model = self._extract_tree_model_from_primary()
        if self.shap_model is not None:
            print("Using embedded RF sub-model for explanation data")
        else:
            self._load_shap_model()

    def _load_shap_model(self):
        """Load the tree-based sub-model used for SHAP explanations."""
        if self.shap_model is not None:
            return

        embedded_model = self._extract_tree_model_from_primary()
        if embedded_model is not None:
            self.shap_model = embedded_model
            return

        for candidate, meta_path in self._shap_model_candidates():
            if not candidate.exists():
                continue
            meta = self._load_metadata(meta_path)
            if not self._is_runtime_compatible(meta):
                continue
            try:
                expected = self._expected_hash(meta, "shap_model_sha256")
                self.shap_model = self._load_pickle_with_integrity(str(candidate), expected)
                self.shap_model_path = candidate
                print(f"SHAP tree model loaded successfully from {candidate}")
                return
            except Exception as exc:
                print(f"Could not load SHAP model from {candidate}: {exc}")
        self.shap_model = None

    def _retrain_model(self):
        """Retrain the model from the training dataset and persist a fresh pickle."""
        from .train_model import train_stress_model
        self.model = train_stress_model()
        self.shap_model = self._extract_tree_model_from_primary()
        if self.shap_model is None:
            self._load_shap_model()

    @staticmethod
    def _build_feature_frame(responses: List[int]) -> pd.DataFrame:
        return pd.DataFrame([responses], columns=EXPECTED_FEATURE_COLUMNS)

    def _stress_level_from_average(self, average_score: float) -> Tuple[int, str]:
        if average_score < 2:
            level = 0
        elif average_score < 3:
            level = 1
        elif average_score < 4:
            level = 2
        else:
            level = 3
        return level, self.stress_labels[level]

    def _compute_weighted_assessment(self, responses: List[int]) -> Dict[str, Any]:
        question_keys = EXPECTED_FEATURE_COLUMNS
        weight_values = np.array([QUESTION_WEIGHTS[q] for q in question_keys], dtype=float)
        response_values = np.array(responses, dtype=float)

        weighted_average = float(np.average(response_values, weights=weight_values))
        weighted_score = float(((weighted_average - 1.0) / 4.0) * 100.0)
        weighted_level, weighted_label = self._stress_level_from_average(weighted_average)

        normalized_stress = np.clip((response_values - 1.0) / 4.0, 0.0, 1.0)
        contribution_values = weight_values * normalized_stress
        total_contribution = float(np.sum(contribution_values))

        top_weighted_questions = []
        for index, question_key in enumerate(question_keys):
            contribution = float(contribution_values[index])
            contribution_percent = (
                (contribution / total_contribution) * 100.0
                if total_contribution > 0
                else 0.0
            )
            top_weighted_questions.append(
                {
                    "question": question_key,
                    "label": QUESTION_LABELS.get(question_key, question_key),
                    "response_value": int(response_values[index]),
                    "weight": round(float(weight_values[index]), 3),
                    "weighted_response": round(float(response_values[index] * weight_values[index]), 3),
                    "stress_contribution": round(contribution, 4),
                    "contribution_percent": round(contribution_percent, 2),
                }
            )

        top_weighted_questions.sort(
            key=lambda item: (item["stress_contribution"], item["weight"], item["response_value"]),
            reverse=True,
        )

        return {
            "average": round(weighted_average, 2),
            "score": round(weighted_score, 1),
            "stress_level": weighted_level,
            "stress_label": weighted_label,
            "question_weights": {
                question: round(float(weight), 3)
                for question, weight in QUESTION_WEIGHTS.items()
            },
            "top_weighted_questions": top_weighted_questions[:6],
        }

    def predict(self, responses: List[int]) -> Tuple[int, str, float, List[str]]:
        """
        Predict stress level from questionnaire responses.
        Returns:
            Tuple of (stress_level, stress_label, confidence, recommendations)
        """
        model = self.model
        if model is None:
            raise Exception("Model not loaded. Please train the model first.")

        if len(responses) != 18:
            raise ValueError(f"Expected 18 responses, got {len(responses)}")

        if not all(1 <= r <= 5 for r in responses):
            raise ValueError("All responses must be between 1 and 5")

        X_raw = self._build_feature_frame(responses)
        X_model = apply_question_weights(X_raw)

        prediction = int(model.predict(X_model)[0])
        probabilities = model.predict_proba(X_model)[0]
        confidence = float(probabilities[prediction])

        stress_label = self.stress_labels[prediction]
        recommendations = self.get_recommendations(prediction, responses)

        return prediction, stress_label, confidence, recommendations

    def predict_with_explanation(self, responses: List[int]) -> Dict[str, Any]:
        """
        Full prediction with SHAP-based explanation.
        Returns a dict with prediction, explainability data, risk factors, and continuous score.
        """
        prediction, stress_label, confidence, recommendations = self.predict(responses)
        model = self.model
        if model is None:
            raise Exception("Model not loaded. Please train the model first.")

        X_raw = self._build_feature_frame(responses)
        X_model = apply_question_weights(X_raw)
        probabilities = model.predict_proba(X_model)[0]

        # --- Continuous stress score (0-100) ---
        # Weighted sum of class probabilities: Low=0, Moderate=33, High=66, Severe=100
        weights = np.array([0, 33.3, 66.6, 100.0])
        continuous_score = float(np.dot(probabilities, weights))

        # --- SHAP explainability ---
        shap_explanation = self._compute_shap(X_model, X_raw, prediction)

        # --- Category-level analysis ---
        category_scores = self._compute_category_scores(responses)

        # --- Weighted questionnaire analysis ---
        weighted_assessment = self._compute_weighted_assessment(responses)

        # --- Risk factors ---
        risk_factors = self._identify_risk_factors(responses, shap_explanation)

        return {
            "stress_level": prediction,
            "stress_label": stress_label,
            "confidence": confidence,
            "continuous_score": round(continuous_score, 1),
            "probabilities": {
                self.stress_labels[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
            "recommendations": recommendations,
            "explanation": shap_explanation,
            "category_scores": category_scores,
            "weighted_assessment": weighted_assessment,
            "risk_factors": risk_factors,
        }

    def _compute_shap(
        self,
        X_model: pd.DataFrame,
        X_raw: pd.DataFrame,
        predicted_class: int,
    ) -> Dict[str, Any]:
        """Compute SHAP values for a single prediction."""
        try:
            shap_module = importlib.import_module("shap")
        except Exception:
            return self._fallback_importance(X_raw)

        tree_model = self.shap_model if self.shap_model is not None else None
        if tree_model is None:
            tree_model = self._extract_tree_model_from_primary()
            if tree_model is not None:
                self.shap_model = tree_model
            else:
                self._load_shap_model()
                tree_model = self.shap_model
        if tree_model is None:
            return self._fallback_importance(X_raw)

        try:
            if self.shap_explainer is None:
                self.shap_explainer = shap_module.TreeExplainer(tree_model)

            shap_values = self.shap_explainer.shap_values(X_model)

            # shap_values shape: (n_classes, n_samples, n_features) or list
            if isinstance(shap_values, list):
                class_shap = shap_values[predicted_class][0]
            else:
                class_shap = shap_values[0] if shap_values.ndim == 2 else shap_values[predicted_class][0]

            feature_names = EXPECTED_FEATURE_COLUMNS
            sorted_idx = np.argsort(np.abs(class_shap))[::-1]

            top_factors = []
            for idx in sorted_idx[:6]:
                fname = feature_names[idx]
                top_factors.append({
                    "question": fname,
                    "label": QUESTION_LABELS.get(fname, fname),
                    "shap_value": round(float(class_shap[idx]), 4),
                    "response_value": int(X_raw.iloc[0][fname]),
                    "impact": "increases_stress" if class_shap[idx] > 0 else "decreases_stress",
                })

            return {
                "method": "shap",
                "top_factors": top_factors,
                "all_shap_values": {
                    feature_names[i]: round(float(class_shap[i]), 4) for i in range(18)
                },
            }
        except Exception as exc:
            print(f"SHAP computation failed: {exc}")
            return self._fallback_importance(X_raw)

    def _fallback_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Use model-level feature importance when SHAP is unavailable."""
        tree_model = self.shap_model if self.shap_model is not None else self._extract_tree_model_from_primary()
        if tree_model is None or not hasattr(tree_model, "feature_importances_"):
            return {"method": "none", "top_factors": [], "all_shap_values": {}}

        importances = tree_model.feature_importances_
        feature_names = EXPECTED_FEATURE_COLUMNS
        sorted_idx = np.argsort(importances)[::-1]

        top_factors = []
        for idx in sorted_idx[:6]:
            fname = feature_names[idx]
            top_factors.append({
                "question": fname,
                "label": QUESTION_LABELS.get(fname, fname),
                "importance": round(float(importances[idx]), 4),
                "response_value": int(X.iloc[0][fname]),
                "impact": "high_response" if X.iloc[0][fname] >= 4 else "normal",
            })

        return {"method": "feature_importance", "top_factors": top_factors, "all_shap_values": {}}

    def _compute_category_scores(self, responses: List[int]) -> Dict[str, Any]:
        """Compute average score per question category."""
        result = {}
        for cat, questions in QUESTION_CATEGORIES.items():
            indices = [int(q[1:]) - 1 for q in questions]
            values = [responses[i] for i in indices]
            avg = sum(values) / len(values)
            # severity: 1-2 low, 2-3 moderate, 3-4 high, 4-5 severe
            if avg < 2:
                severity = "low"
            elif avg < 3:
                severity = "moderate"
            elif avg < 4:
                severity = "high"
            else:
                severity = "severe"
            result[cat] = {"average": round(avg, 2), "severity": severity, "scores": values}
        return result

    def _identify_risk_factors(self, responses: List[int], shap_data: Dict) -> List[Dict[str, Any]]:
        """Identify clinically significant risk factors from responses."""
        risks = []
        # Sleep issues
        if responses[5] >= 4:
            risks.append({"factor": "sleep_disruption", "severity": "high",
                          "question": "q6", "label": "Sleep trouble",
                          "message": "Significant sleep disruption detected. This strongly correlates with elevated stress."})
        # Suicidal ideation proxy (severe negative thoughts + social withdrawal + overwhelm)
        if responses[8] >= 4 and responses[12] >= 4 and responses[13] >= 4:
            risks.append({"factor": "combined_withdrawal", "severity": "critical",
                          "label": "Withdrawal + Negative thoughts",
                          "message": "Combination of negative thoughts, social withdrawal, and overwhelm detected. Professional assessment recommended."})
        # Cardiovascular stress
        if responses[6] >= 4:
            risks.append({"factor": "cardiovascular_stress", "severity": "high",
                          "question": "q7", "label": "Rapid heartbeat/chest tightness",
                          "message": "Physical stress symptoms (heart/chest) detected. Consider medical evaluation."})
        # Financial + work compound stress
        if responses[15] >= 4 and responses[17] >= 4:
            risks.append({"factor": "compound_external_stress", "severity": "moderate",
                          "question": "q16+q18", "label": "Work + Financial stress",
                          "message": "Both work/study and financial stressors are high — a compound stress pattern."})
        # Overall high average
        avg = sum(responses) / len(responses)
        if avg >= 4.0:
            risks.append({"factor": "global_high_stress", "severity": "critical",
                          "label": "All domains elevated",
                          "message": "Nearly all stress indicators are elevated. Immediate professional help is recommended."})
        return risks

    def get_recommendations(self, stress_level: int, responses: List[int]) -> List[str]:
        """Generate personalized recommendations based on stress level."""
        if stress_level == 0:
            recommendations = [
                "Your stress levels appear well managed. Keep your current self-care routine.",
                "Maintain regular exercise and healthy eating habits.",
                "Keep spending time on activities you enjoy.",
                "Use preventive stress-management techniques such as meditation.",
            ]
        elif stress_level == 1:
            recommendations = [
                "You are experiencing moderate stress. Take early action now.",
                "Practice daily relaxation techniques such as breathing or meditation.",
                "Aim for 7 to 8 hours of quality sleep.",
                "Try regular physical activity for at least 30 minutes a day.",
                "Talk with friends, family, or a counselor about your concerns.",
            ]
        elif stress_level == 2:
            recommendations = [
                "You are experiencing high stress. Professional support is recommended.",
                "Consider scheduling an appointment with a mental health professional.",
                "Practice stress-reduction techniques multiple times a day.",
                "Prioritize and organize tasks to reduce overwhelm.",
                "Limit caffeine and alcohol intake.",
                "Reach out to your support network promptly.",
            ]
        else:
            recommendations = [
                "Urgent: you are experiencing severe stress. Seek professional help immediately.",
                "Book an appointment with a doctor or mental health professional today.",
                "Contact a crisis helpline if you are in immediate distress.",
                "Consider speaking with a psychiatrist about your symptoms.",
                "Tell trusted family members or friends how you are feeling.",
                "Step away from stressful activities if possible.",
            ]

        if responses[5] >= 4:
            recommendations.append(
                "Focus on improving sleep hygiene and keeping a regular sleep schedule."
            )
        if responses[2] >= 4:
            recommendations.append(
                "Practice relaxation techniques when you feel irritable or angry."
            )
        if responses[12] >= 4:
            recommendations.append(
                "Try to maintain social connection, even if only briefly."
            )
        if responses[15] >= 4:
            recommendations.append(
                "Review your work-life balance and set healthier boundaries."
            )

        return recommendations

    def get_stress_trend(self, test_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyse a user's test history for stress trends.
        Uses linear regression on the continuous score to detect direction.
        """
        if len(test_history) < 2:
            return {"trend": "insufficient_data", "tests_analysed": len(test_history)}

        scores = []
        ordered_history = sorted(
            test_history,
            key=lambda t: t.get("timestamp") or 0,
        )
        for t in ordered_history:
            level = t.get("stress_level", 0)
            scores.append(level)

        # Linear trend via numpy polyfit
        x = np.arange(len(scores), dtype=float)
        y = np.array(scores, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)

        if slope > 0.15:
            trend = "worsening"
        elif slope < -0.15:
            trend = "improving"
        else:
            trend = "stable"

        # Volatility (std dev)
        volatility = float(np.std(y))

        # Moving average (last 3)
        recent_avg = float(np.mean(y[-min(3, len(y)):]))

        # Predict next stress level
        predicted_next = float(slope * len(scores) + intercept)
        predicted_next = max(0, min(3, predicted_next))

        return {
            "trend": trend,
            "slope": round(float(slope), 4),
            "volatility": round(volatility, 4),
            "recent_average": round(recent_avg, 2),
            "predicted_next_level": round(predicted_next, 2),
            "forecast": stress_forecaster.forecast_levels(scores, horizon=3),
            "tests_analysed": len(scores),
            "history": [
                {"stress_level": int(scores[i]), "index": i}
                for i in range(len(scores))
            ],
        }

    def check_crisis(self, user_id: str, test_history: List[Dict], current_result: Dict) -> Dict[str, Any]:
        """
        Evaluate whether the user is in a crisis state.
        Criteria:
          - Current test is Severe (level 3)
          - 3+ consecutive Severe tests
          - Sharp spike from previous test
        """
        is_crisis = False
        crisis_reasons = []
        severity = "none"

        current_level = current_result.get("stress_level", 0)

        # Check if current is Severe
        if current_level == 3:
            crisis_reasons.append("Current assessment indicates severe stress")
            severity = "high"

        # Check for 3+ consecutive severe
        if len(test_history) >= 2:
            recent = [t.get("stress_level", 0) for t in test_history[:3]]
            recent.insert(0, current_level)
            consecutive_severe = 0
            for level in recent:
                if level >= 3:
                    consecutive_severe += 1
                else:
                    break
            if consecutive_severe >= 3:
                is_crisis = True
                severity = "critical"
                crisis_reasons.append(f"{consecutive_severe} consecutive severe stress assessments")

        # Check for sudden spike
        if test_history and current_level >= 2:
            prev_level = test_history[0].get("stress_level", 0)
            if current_level - prev_level >= 2:
                crisis_reasons.append(f"Sudden stress spike: {prev_level} → {current_level}")
                if current_level == 3:
                    is_crisis = True
                    severity = "critical"
                else:
                    severity = max(severity, "high") if severity != "critical" else severity

        if current_level == 3 and len(crisis_reasons) >= 2:
            is_crisis = True
            severity = "critical"

        return {
            "is_crisis": is_crisis,
            "severity": severity,
            "reasons": crisis_reasons,
            "recommended_actions": self._crisis_actions(severity) if crisis_reasons else [],
        }

    def _crisis_actions(self, severity: str) -> List[Dict[str, str]]:
        if severity == "critical":
            return [
                {"action": "contact_crisis_line", "message": "Please contact a crisis helpline immediately", "priority": "urgent"},
                {"action": "notify_doctor", "message": "Your assigned doctor will be notified", "priority": "urgent"},
                {"action": "emergency_contact", "message": "Consider reaching out to your emergency contact", "priority": "high"},
            ]
        elif severity == "high":
            return [
                {"action": "book_appointment", "message": "Schedule an appointment with a mental health professional", "priority": "high"},
                {"action": "use_coping", "message": "Use immediate coping techniques (breathing, grounding)", "priority": "medium"},
            ]
        return []

    def compute_sentiment_scores(self, messages: List[str]) -> Dict[str, Any]:
        """
        Basic NLP sentiment analysis on chatbot messages.
        Uses keyword-based scoring when full NLP models are unavailable.
        """
        negative_words = {
            "sad", "depressed", "hopeless", "anxious", "worried", "scared", "angry",
            "frustrated", "overwhelmed", "exhausted", "tired", "lonely", "worthless",
            "helpless", "panic", "fear", "stress", "crying", "hurt", "pain", "suffer",
            "terrible", "awful", "nightmare", "can't cope", "giving up", "no point",
            "hate", "miserable", "desperate", "suicidal", "kill", "die", "end it",
        }
        positive_words = {
            "happy", "grateful", "hopeful", "calm", "relaxed", "better", "good",
            "great", "wonderful", "peaceful", "content", "joy", "progress", "improve",
            "strong", "confident", "motivated", "optimistic", "love", "thankful",
        }
        crisis_words = {"suicidal", "kill myself", "end it all", "want to die", "no reason to live", "self-harm"}

        total_neg = 0
        total_pos = 0
        msg_count = max(len(messages), 1)
        crisis_detected = False
        crisis_messages = []

        for msg in messages:
            lower = msg.lower()
            for cw in crisis_words:
                if cw in lower:
                    crisis_detected = True
                    crisis_messages.append(msg)
                    break
            words = set(lower.split())
            neg = len(words & negative_words)
            pos = len(words & positive_words)
            total_neg += neg
            total_pos += pos

        sentiment_score = (total_pos - total_neg) / msg_count
        if sentiment_score < -1:
            mood = "very_negative"
        elif sentiment_score < 0:
            mood = "negative"
        elif sentiment_score < 0.5:
            mood = "neutral"
        else:
            mood = "positive"

        return {
            "sentiment_score": round(sentiment_score, 3),
            "mood": mood,
            "positive_count": total_pos,
            "negative_count": total_neg,
            "crisis_detected": crisis_detected,
            "crisis_messages": crisis_messages[:3] if crisis_detected else [],
            "messages_analysed": len(messages),
        }

    def retrain_with_new_data(self, new_responses: List[List[int]], new_labels: List[int]):
        """Append validated labeled rows to the dataset CSV and retrain the questionnaire model."""
        if not new_responses or not new_labels:
            raise ValueError("Provide at least one labeled sample for retraining.")
        if len(new_responses) != len(new_labels):
            raise ValueError("new_responses and new_labels must have the same length.")

        feature_columns = [f"q{i+1}" for i in range(18)]
        validated_rows = []
        for responses, label in zip(new_responses, new_labels):
            if len(responses) != 18:
                raise ValueError(f"Each response set must contain 18 answers, got {len(responses)}.")
            if not all(1 <= int(response) <= 5 for response in responses):
                raise ValueError("All questionnaire responses must be integers from 1 to 5.")

            normalized_label = int(label)
            if normalized_label not in self.stress_labels:
                raise ValueError(f"Unsupported stress label '{label}'. Expected one of {list(self.stress_labels)}.")

            validated_rows.append([int(response) for response in responses] + [normalized_label])

        dataset_path = os.path.join(os.path.dirname(__file__), "stress_training_dataset_100k.csv")
        expected_columns = feature_columns + ["stress_level"]
        incoming_df = pd.DataFrame(validated_rows, columns=expected_columns)

        if os.path.exists(dataset_path):
            existing_df = pd.read_csv(dataset_path)
            missing_columns = [column for column in expected_columns if column not in existing_df.columns]
            if missing_columns:
                raise ValueError(f"Existing training dataset is missing required columns: {missing_columns}")
            combined_df = pd.concat([existing_df[expected_columns], incoming_df], ignore_index=True)
        else:
            combined_df = incoming_df

        combined_df.to_csv(dataset_path, index=False)
        self._retrain_model()

        return {
            "status": "retrained",
            "added_rows": int(len(incoming_df)),
            "total_rows": int(len(combined_df)),
            "dataset_path": dataset_path,
        }


predictor = StressPredictor()
