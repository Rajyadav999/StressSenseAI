#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║        STRESS LEVEL DETECTION SYSTEM - MindScan Pro          ║
║   Voice-Based | Psychological + Behavioural + Cognitive      ║
║   Scales: PSS-10, GAD-7, PHQ-9, Perceived Stress Index       ║
╚══════════════════════════════════════════════════════════════╝
"""

import sqlite3
import json
import http.server
import threading
import webbrowser
import os
import base64
import io
import datetime
import math
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

APP_DIR = Path(__file__).resolve().parent
ML_DATASET_PATH = APP_DIR / "ml_model" / "stress_training_dataset_100k.csv"
ML_MODEL_LOCK = threading.Lock()
ML_MODEL_BUNDLE = None
ML_FEATURE_COLUMNS = [f"q{i}" for i in range(1, 19)]
ML_STRESS_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Severe",
}
ML_FEATURE_LABELS = {
    "q1": "Nervous or anxious feelings",
    "q2": "Low mood or depression",
    "q3": "Irritable or angry mood",
    "q4": "Headaches or body pain",
    "q5": "Physical fatigue",
    "q6": "Sleep trouble",
    "q7": "Rapid heartbeat or chest tension",
    "q8": "Difficulty concentrating",
    "q9": "Negative thoughts",
    "q10": "Worry about the future",
    "q11": "Difficulty making decisions",
    "q12": "Appetite changes",
    "q13": "Avoiding social contact",
    "q14": "Feeling overwhelmed by tasks",
    "q15": "Work-life balance strain",
    "q16": "Work or study stress",
    "q17": "Relationship stress",
    "q18": "Financial stress pressure",
}


def _answer_score_1_to_5(answers: dict, question_id: int, default: int = 3) -> int:
    raw_value = answers.get(str(question_id))
    if raw_value is None:
        return default

    try:
        score = int(raw_value) + 1
    except (TypeError, ValueError):
        return default
    return max(1, min(5, score))


def _average_likert(values, default: int = 3) -> int:
    clean_values = [int(value) for value in values if value is not None]
    if not clean_values:
        return default

    average = sum(clean_values) / len(clean_values)
    return max(1, min(5, int(math.floor(average + 0.5))))


def _map_answers_to_ml_features(answers: dict) -> dict:
    score = lambda qid, default=3: _answer_score_1_to_5(answers, qid, default=default)

    # Map the app's 18 assessment questions into the questionnaire model's
    # 18 feature slots using the closest available signals. This avoids
    # silently falling back to defaults for non-existent question IDs.
    mapped = {
        "q1": _average_likert([score(2), score(3)]),
        "q2": score(6),
        "q3": _average_likert([score(3), score(5), score(16)]),
        "q4": score(11),
        "q5": _average_likert([score(8), score(11)]),
        "q6": score(8),
        "q7": score(11),
        "q8": score(15),
        "q9": _average_likert([score(5), score(6), score(16)]),
        "q10": _average_likert([score(1), score(3), score(4), score(5)]),
        "q11": score(17),
        "q12": score(12),
        "q13": score(10),
        "q14": _average_likert([score(4), score(13), score(15)]),
        "q15": _average_likert([score(1), score(13), score(14)]),
        "q16": _average_likert([score(1), score(4), score(13), score(15)]),
        "q17": _average_likert([score(2), score(5), score(10)]),
        "q18": _average_likert([score(1), score(4), score(13)]),
    }

    return {column: mapped[column] for column in ML_FEATURE_COLUMNS}


def get_ml_model_bundle():
    global ML_MODEL_BUNDLE

    if ML_MODEL_BUNDLE is not None and ML_MODEL_BUNDLE.get("available"):
        return ML_MODEL_BUNDLE

    with ML_MODEL_LOCK:
        if ML_MODEL_BUNDLE is not None and ML_MODEL_BUNDLE.get("available"):
            return ML_MODEL_BUNDLE

        try:
            from ml_model.predictor import predictor as questionnaire_predictor

            ML_MODEL_BUNDLE = {
                "available": questionnaire_predictor.model is not None,
                "predictor": questionnaire_predictor,
                "model": questionnaire_predictor.model,
                "dataset_path": None,
                "dataset_rows": 0,
                "validation_accuracy": 0.0,
                "class_distribution": {},
                "model_type": "unknown",
                "reason": "",
            }

            meta_path = getattr(questionnaire_predictor, "meta_path", None)
            if meta_path and Path(meta_path).exists():
                with open(meta_path, "r", encoding="utf-8") as meta_file:
                    metadata = json.load(meta_file)
                ML_MODEL_BUNDLE["dataset_path"] = metadata.get("dataset_path")
                ML_MODEL_BUNDLE["dataset_rows"] = int(metadata.get("total_rows") or 0)
                ML_MODEL_BUNDLE["validation_accuracy"] = float(
                    metadata.get("accuracy") or metadata.get("cv_accuracy_mean") or 0.0
                )
                ML_MODEL_BUNDLE["class_distribution"] = metadata.get("class_distribution", {})
                ML_MODEL_BUNDLE["model_type"] = metadata.get("model_type", "unknown")

            if not ML_MODEL_BUNDLE["dataset_path"] and ML_DATASET_PATH.exists():
                ML_MODEL_BUNDLE["dataset_path"] = str(ML_DATASET_PATH)

            if not ML_MODEL_BUNDLE["available"]:
                ML_MODEL_BUNDLE["reason"] = "Questionnaire model could not be loaded."
            else:
                print(
                    "Integrated questionnaire ML model ready: "
                    f"{ML_MODEL_BUNDLE['model_type']} "
                    f"(accuracy {ML_MODEL_BUNDLE['validation_accuracy']:.4f})"
                )
        except Exception as exc:
            ML_MODEL_BUNDLE = {
                "available": False,
                "reason": str(exc),
            }
            print(f"Dataset-backed ML integration disabled: {exc}")

    return ML_MODEL_BUNDLE


def get_ml_assessment(answers: dict):
    bundle = get_ml_model_bundle()
    if not bundle.get("available"):
        return None

    try:
        mapped_features = _map_answers_to_ml_features(answers)
        predictor = bundle["predictor"]
        ordered_responses = [mapped_features[column] for column in ML_FEATURE_COLUMNS]
        prediction = predictor.predict_with_explanation(ordered_responses)

        top_signals_source = (
            prediction.get("weighted_assessment", {}).get("top_weighted_questions", [])
            or prediction.get("explanation", {}).get("top_factors", [])
        )
        top_signals = []
        for signal in top_signals_source[:5]:
            question_key = signal.get("question")
            top_signals.append(
                {
                    "question": question_key,
                    "label": ML_FEATURE_LABELS.get(question_key, signal.get("label", question_key)),
                    "score": int(signal.get("response_value", mapped_features.get(question_key, 0))),
                }
            )

        if not top_signals:
            top_signals = [
                {
                    "question": question_key,
                    "label": ML_FEATURE_LABELS.get(question_key, question_key),
                    "score": int(score),
                }
                for question_key, score in sorted(
                    mapped_features.items(),
                    key=lambda item: (item[1], item[0]),
                    reverse=True,
                )[:5]
            ]

        validation_accuracy = float(bundle.get("validation_accuracy") or 0.0)

        return {
            "stress_level": int(prediction["stress_level"]),
            "stress_label": prediction["stress_label"],
            "confidence": round(float(prediction["confidence"]), 4),
            "confidence_pct": round(float(prediction["confidence"]) * 100, 1),
            "continuous_score": round(float(prediction["continuous_score"]), 1),
            "dataset_rows": int(bundle["dataset_rows"]),
            "validation_accuracy": round(validation_accuracy, 4),
            "validation_accuracy_pct": round(validation_accuracy * 100, 1),
            "class_distribution": bundle["class_distribution"],
            "dataset_path": bundle["dataset_path"],
            "model_type": bundle.get("model_type", "unknown"),
            "mapping_note": (
                "Derived from the current 18-question assessment using the closest "
                "18-questionnaire-feature proxy mapping."
            ),
            "mapped_features": mapped_features,
            "top_signals": top_signals,
            "probabilities": {
                label: round(float(probability) * 100, 1)
                for label, probability in prediction.get("probabilities", {}).items()
            },
            "weighted_assessment": prediction.get("weighted_assessment", {}),
            "risk_factors": prediction.get("risk_factors", []),
            "explanation_method": prediction.get("explanation", {}).get("method", "none"),
        }
    except Exception as exc:
        print(f"Dataset-backed ML assessment failed: {exc}")
        return None

def init_db():
    conn = sqlite3.connect("stress_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        answers TEXT,
        psychological REAL,
        behavioural REAL,
        cognitive REAL,
        composite REAL,
        level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def save_to_db(name, answers, scores):
    conn = sqlite3.connect("stress_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO responses
    (name, answers, psychological, behavioural, cognitive, composite, level)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        json.dumps(answers),
        scores["psychological_pct"],
        scores["behavioural_pct"],
        scores["cognitive_pct"],
        scores["composite"],
        scores["level"]
    ))

    conn.commit()
    conn.close()

def get_all_responses():
    conn = sqlite3.connect("stress_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, level, composite, created_at FROM responses ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    responses = []
    for row in rows:
        responses.append({
            "id": row[0],
            "name": row[1],
            "level": row[2],
            "composite": row[3],
            "created_at": row[4]
        })
    return responses

# ─────────────────────────────────────────────────────────────
#  QUESTION BANK
#  Categories: P=Psychological(×1.5), B=Behavioural(×1.3), C=Cognitive(×1.0)
#  Scale: 0=Never, 1=Rarely, 2=Sometimes, 3=Often, 4=Always
# ─────────────────────────────────────────────────────────────

QUESTIONS = [
    # ── PSYCHOLOGICAL (weight 1.5) ──────────────────────────────
    {
        "id": 1, "category": "P", "weight": 1.5,
        "scale": "PSS-10",
        "text": "How often have you felt that you were unable to control the important things in your life?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 2, "category": "P", "weight": 1.5,
        "scale": "PSS-10",
        "text": "How often have you felt nervous and stressed in the past month?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 3, "category": "P", "weight": 1.5,
        "scale": "GAD-7",
        "text": "How often have you felt overwhelming anxiety or worry that is difficult to control?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 4, "category": "P", "weight": 1.5,
        "scale": "PSS-10",
        "text": "How often have you felt difficulties were piling up so high that you could not overcome them?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 5, "category": "P", "weight": 1.5,
        "scale": "GAD-7",
        "text": "How often do you feel a sense of dread or impending doom without a clear reason?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 6, "category": "P", "weight": 1.5,
        "scale": "PHQ-9",
        "text": "How often have you felt little interest or pleasure in doing things you normally enjoy?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 7, "category": "P", "weight": 1.5,
        "scale": "PSS-10",
        "text": "How often have you felt that things were going your way? (Reverse scored)",
        "options": ["Always", "Often", "Sometimes", "Rarely", "Never"],
        "reversed": True
    },

    # ── BEHAVIOURAL (weight 1.3) ────────────────────────────────
    {
        "id": 8, "category": "B", "weight": 1.3,
        "scale": "DASS-21",
        "text": "How often do you experience disturbed sleep — difficulty falling or staying asleep?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 9, "category": "B", "weight": 1.3,
        "scale": "DASS-21",
        "text": "How often do you rely on substances (caffeine, alcohol, or other) to manage your mood or energy?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 10, "category": "B", "weight": 1.3,
        "scale": "ISI",
        "text": "How often do you withdraw from social activities or avoid interacting with others?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 11, "category": "B", "weight": 1.3,
        "scale": "DASS-21",
        "text": "How often do you experience physical symptoms of stress — headaches, muscle tension, or upset stomach?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 12, "category": "B", "weight": 1.3,
        "scale": "PHQ-9",
        "text": "How often do you find yourself overeating, under-eating, or making unhealthy food choices under pressure?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 13, "category": "B", "weight": 1.3,
        "scale": "DASS-21",
        "text": "How often do you procrastinate or feel paralysed when faced with tasks or deadlines?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 14, "category": "B", "weight": 1.3,
        "scale": "ISI",
        "text": "How often do you exercise, meditate, or practise other stress-relief activities? (Reverse scored)",
        "options": ["Daily", "Often", "Sometimes", "Rarely", "Never"],
        "reversed": True
    },

    # ── COGNITIVE (weight 1.0) ──────────────────────────────────
    {
        "id": 15, "category": "C", "weight": 1.0,
        "scale": "CDS",
        "text": "How often do you struggle to concentrate or find your mind wandering during important tasks?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 16, "category": "C", "weight": 1.0,
        "scale": "CDS",
        "text": "How often do you experience racing or intrusive thoughts that are difficult to stop?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 17, "category": "C", "weight": 1.0,
        "scale": "CDS",
        "text": "How often do you make more errors or poor decisions than usual?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    {
        "id": 18, "category": "C", "weight": 1.0,
        "scale": "CDS",
        "text": "How often do you experience memory lapses — forgetting appointments, names, or recent events?",
        "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
    },
    
]

# ─────────────────────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────────────────────

def _compute_domain_breakdown(answers: dict) -> dict:
    p_raw, p_max = 0, 0
    b_raw, b_max = 0, 0
    c_raw, c_max = 0, 0

    for q in QUESTIONS:
        qid = str(q["id"])
        if qid not in answers:
            continue
        score = int(answers[qid])
        weighted = score * q["weight"]
        max_weighted = 4 * q["weight"]

        if q["category"] == "P":
            p_raw += weighted
            p_max += max_weighted
        elif q["category"] == "B":
            b_raw += weighted
            b_max += max_weighted
        else:
            c_raw += weighted
            c_max += max_weighted

    def pct(raw, mx):
        return round((raw / mx * 100) if mx > 0 else 0, 1)

    return {
        "psychological_pct": pct(p_raw, p_max),
        "behavioural_pct": pct(b_raw, b_max),
        "cognitive_pct": pct(c_raw, c_max),
        "p_raw": round(p_raw, 2),
        "b_raw": round(b_raw, 2),
        "c_raw": round(c_raw, 2),
    }


def _style_for_ml_label(stress_label: str) -> dict:
    styles = {
        "Low": {"level": "Low", "color": "#2ecc71", "emoji": "🟢", "severity": 0},
        "Moderate": {"level": "Moderate", "color": "#f1c40f", "emoji": "🟡", "severity": 1},
        "High": {"level": "High", "color": "#e67e22", "emoji": "🟠", "severity": 2},
        "Severe": {"level": "Severe", "color": "#e74c3c", "emoji": "🔴", "severity": 3},
    }
    return styles.get(
        stress_label,
        {"level": stress_label, "color": "#7c3aed", "emoji": "⚪", "severity": 0},
    )


def compute_scores(answers: dict) -> dict:
    """answers = {question_id: score_0_to_4}"""
    domain_scores = _compute_domain_breakdown(answers)
    ml_assessment = get_ml_assessment(answers)

    if not ml_assessment:
        bundle = get_ml_model_bundle()
        reason = str(bundle.get("reason") or "Questionnaire ML model is unavailable.").strip()
        raise RuntimeError(f"Questionnaire ML scoring is unavailable: {reason}")

    composite = float(ml_assessment["continuous_score"])
    primary_style = _style_for_ml_label(ml_assessment["stress_label"])
    return {
        "psychological_pct": domain_scores["psychological_pct"],
        "behavioural_pct": domain_scores["behavioural_pct"],
        "cognitive_pct": domain_scores["cognitive_pct"],
        "composite": round(composite, 1),
        "level": primary_style["level"],
        "color": primary_style["color"],
        "emoji": primary_style["emoji"],
        "severity": primary_style["severity"],
        "p_raw": domain_scores["p_raw"],
        "b_raw": domain_scores["b_raw"],
        "c_raw": domain_scores["c_raw"],
        "score_engine": "ml_primary",
        "model_label": ml_assessment["stress_label"],
        "model_confidence_pct": ml_assessment["confidence_pct"],
        "ml_assessment": ml_assessment,
    }


def get_suggestions(scores: dict) -> list:
    level = scores["level"]
    p = scores["psychological_pct"]
    b = scores["behavioural_pct"]
    c = scores["cognitive_pct"]

    if level == "Low":
        level = "Minimal"

    suggestions = {
        "Minimal": [
            "🌿 Maintain your current healthy routines — you're doing great!",
            "🧘 Continue mindfulness practices to sustain low stress levels.",
            "📖 Consider journaling to build self-awareness over time.",
            "🏃 Keep up regular physical activity for long-term resilience.",
            "🤝 Stay socially connected — nurture your support network.",
        ],
        "Mild": [
            "⏱️ Practice the 4-7-8 breathing technique: inhale 4s, hold 7s, exhale 8s.",
            "📝 Start a daily gratitude journal — write 3 things each evening.",
            "🚶 Take a 20-minute walk in nature at least 4 times a week.",
            "📵 Reduce screen time 1 hour before bed to improve sleep quality.",
            "🎯 Use time-blocking to manage tasks and reduce overwhelm.",
        ],
        "Moderate": [
            "🧠 Consider Cognitive Behavioural Therapy (CBT) techniques or workshops.",
            "💤 Prioritise sleep hygiene: consistent bedtime, dark/cool room, no caffeine after 2pm.",
            "🏋️ Engage in moderate aerobic exercise (30 min/day, 5 days/week).",
            "👥 Talk to a trusted friend, mentor, or counsellor about your stressors.",
            "📲 Try structured mindfulness apps (Headspace, Calm, Insight Timer).",
            "🍎 Review your nutrition — reduce processed foods and increase omega-3 intake.",
        ],
        "High": [
            "🆘 Strongly consider consulting a licensed psychologist or therapist.",
            "📋 Work with a professional to identify and address root stressors.",
            "💊 Discuss with your doctor whether lifestyle or medical intervention is appropriate.",
            "🛑 Immediately reduce non-essential commitments and set firm boundaries.",
            "🧘 Practise progressive muscle relaxation (PMR) daily — 20 minutes.",
            "📞 Call a stress or mental health helpline if you feel overwhelmed.",
        ],
        "Severe": [
            "🚨 Please seek immediate professional mental health support.",
            "👨‍⚕️ Schedule an urgent appointment with a psychiatrist or psychologist.",
            "📞 India: iCall helpline — 9152987821 | Vandrevala Foundation — 1860-2662-345.",
            "🏥 If you feel unsafe, visit your nearest emergency mental health facility.",
            "💙 Remember: reaching out for help is a sign of strength, not weakness.",
            "🤝 Lean on your immediate support network — family, close friends.",
        ],
    }

    domain_tips = []
    if p > 60:
        domain_tips.append("🧩 Psychological: Practice emotional regulation techniques like RAIN (Recognise, Allow, Investigate, Nurture).")
    if b > 60:
        domain_tips.append("🏃 Behavioural: Establish a consistent sleep-wake schedule and reduce stimulant use.")
    if c > 60:
        domain_tips.append("🧠 Cognitive: Use the Pomodoro technique (25 min focus / 5 min break) to improve concentration.")

    return suggestions.get(level, []) + domain_tips


# ─────────────────────────────────────────────────────────────
#  CHART GENERATORS
# ─────────────────────────────────────────────────────────────

def make_gauge_chart(score: float, level: str, color: str) -> str:
    """Returns base64 PNG of a gauge/speedometer chart."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    # Draw gauge arcs
    zones = [
        (0, 20,   '#2ecc71', 'Minimal'),
        (20, 40,  '#f1c40f', 'Mild'),
        (40, 60,  '#e67e22', 'Moderate'),
        (60, 80,  '#e74c3c', 'High'),
        (80, 100, '#8e44ad', 'Severe'),
    ]

    theta_start = math.pi  # 180°
    theta_end   = 0        # 0°

    for lo, hi, zcolor, _ in zones:
        t0 = theta_start - (lo / 100) * math.pi
        t1 = theta_start - (hi / 100) * math.pi
        theta = np.linspace(t0, t1, 50)
        r_inner, r_outer = 0.55, 0.9
        xs = np.concatenate([r_outer*np.cos(theta), r_inner*np.cos(theta[::-1])])
        ys = np.concatenate([r_outer*np.sin(theta), r_inner*np.sin(theta[::-1])])
        ax.fill(xs, ys, color=zcolor, alpha=0.85, zorder=2)

    # Needle
    needle_angle = math.pi - (score / 100) * math.pi
    nx = 0.75 * math.cos(needle_angle)
    ny = 0.75 * math.sin(needle_angle)
    ax.annotate('', xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
    ax.add_patch(plt.Circle((0, 0), 0.05, color='white', zorder=5))

    # Score text
    ax.text(0, -0.25, f"{score:.1f}%", ha='center', va='center',
            fontsize=22, fontweight='bold', color='white', fontfamily='monospace')
    ax.text(0, -0.42, level, ha='center', va='center',
            fontsize=13, color=color, fontweight='bold')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.6, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Composite Stress Score', color='#aaa', fontsize=11, pad=8)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_radar_chart(p: float, b: float, c: float) -> str:
    """Radar/spider chart for domain scores."""
    categories = ['Psychological\n(×1.5)', 'Behavioural\n(×1.3)', 'Cognitive\n(×1.0)']
    values = [p, b, c]
    N = 3
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor='#0d1117')
    ax.set_facecolor('#161b22')

    # Grid
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#555', fontsize=7)
    ax.yaxis.set_tick_params(pad=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='#ccc', fontsize=9)
    ax.grid(color='#333', linestyle='--', linewidth=0.5)
    ax.spines['polar'].set_color('#333')

    # Fill
    zone_colors = ['#2ecc71','#f1c40f','#e67e22','#e74c3c','#8e44ad']
    avg = (p + b + c) / 3
    zidx = min(int(avg / 20), 4)
    fill_color = zone_colors[zidx]

    ax.plot(angles, values_plot, color=fill_color, linewidth=2, linestyle='solid')
    ax.fill(angles, values_plot, color=fill_color, alpha=0.25)

    # Dots
    for angle, val, cat in zip(angles[:-1], values, ['Psychological', 'Behavioural', 'Cognitive']):
        ax.plot(angle, val, 'o', color=fill_color, markersize=8, zorder=5)

    ax.set_title('Domain Stress Profile', color='#aaa', fontsize=11, pad=20)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_bar_chart(p: float, b: float, c: float) -> str:
    """Horizontal bar chart comparing domains."""
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0d1117')
    ax.set_facecolor('#161b22')

    domains = ['Psychological\n(×1.5)', 'Behavioural\n(×1.3)', 'Cognitive\n(×1.0)']
    values = [p, b, c]
    bar_colors = ['#e74c3c', '#e67e22', '#3498db']

    bars = ax.barh(domains, values, color=bar_colors, height=0.5, alpha=0.85)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', color='white', fontsize=10, fontweight='bold')

    # Reference line at 50%
    ax.axvline(50, color='#555', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(51, -0.6, 'Moderate\nthreshold', color='#666', fontsize=7)

    ax.set_xlim(0, 110)
    ax.set_xlabel('Stress Score (%)', color='#aaa', fontsize=9)
    ax.tick_params(colors='#aaa', labelsize=9)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Domain Breakdown', color='#aaa', fontsize=11, pad=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────
#  PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_pdf_report(scores: dict, answers: dict, user_name: str = "User") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold')

    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#555'),
        spaceAfter=20, alignment=TA_CENTER)

    heading_style = ParagraphStyle('H', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor('#1a1a2e'),
        spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold')

    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#333'),
        spaceAfter=5, leading=15, alignment=TA_JUSTIFY)

    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#333'),
        spaceAfter=4, leading=14, leftIndent=12)

    # ── Header
    story.append(Paragraph("MindScan Pro", title_style))
    story.append(Paragraph("Stress Level Detection Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
    story.append(Spacer(1, 0.4*cm))

    # ── Meta info
    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    meta = [
        ["Participant", user_name],
        ["Date & Time", now],
        ["Assessment", "ML-Based Psychological Stress Evaluation"],
        ["Scales Used", "PSS-10 · GAD-7 · PHQ-9 · DASS-21 · ISI · CDS"],
    ]
    t = Table(meta, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f0f4ff')),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ddd')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#fafafa')]),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#333')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Score Summary Table
    story.append(Paragraph("ML Score Summary", heading_style))

    level_colors = {
        'Low': '#2ecc71', 'Minimal': '#2ecc71',
        'Mild': '#f1c40f', 'Moderate': '#f1c40f',
        'High': '#e67e22', 'Severe': '#e74c3c'
    }
    lc = level_colors.get(scores['level'], '#333')
    ml_assessment = scores.get("ml_assessment") or get_ml_assessment(answers)

    score_data = [
        ["Domain", "Role", "Score (%)", "Interpretation"],
        ["Psychological", "×1.5", f"{scores['psychological_pct']}%", _interpret(scores['psychological_pct'])],
        ["Behavioural", "×1.3", f"{scores['behavioural_pct']}%", _interpret(scores['behavioural_pct'])],
        ["Cognitive", "×1.0", f"{scores['cognitive_pct']}%", _interpret(scores['cognitive_pct'])],
        ["COMPOSITE SCORE", "Weighted", f"{scores['composite']}%", scores['level']],
    ]
    score_data = [
        ["Domain", "Role", "Score (%)", "Interpretation"],
        ["Psychological", "Supporting", f"{scores['psychological_pct']}%", _interpret(scores['psychological_pct'])],
        ["Behavioural", "Supporting", f"{scores['behavioural_pct']}%", _interpret(scores['behavioural_pct'])],
        ["Cognitive", "Supporting", f"{scores['cognitive_pct']}%", _interpret(scores['cognitive_pct'])],
        ["PRIMARY SCORE", "ML Model", f"{scores['composite']}%", scores['level']],
    ]
    st = Table(score_data, colWidths=[5*cm, 2.5*cm, 3*cm, 6.5*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,4), (-1,4), 'Helvetica-Bold'),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor(lc + '33')),
        ('TEXTCOLOR', (2,4), (2,4), colors.HexColor(lc)),
        ('TEXTCOLOR', (3,4), (3,4), colors.HexColor(lc)),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,3), [colors.white, colors.HexColor('#fafafa')]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#333')),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.5*cm))

    if ml_assessment:
        story.append(Paragraph("ML Model Summary", heading_style))
        ml_level_colors = {
            "Low": "#2ecc71",
            "Moderate": "#f1c40f",
            "High": "#e67e22",
            "Severe": "#e74c3c",
        }
        ml_color = ml_level_colors.get(ml_assessment["stress_label"], "#333333")
        ml_data = [
            ["Metric", "Value"],
            ["ML label", ml_assessment["stress_label"]],
            ["Confidence", f"{ml_assessment['confidence_pct']}%"],
            ["Continuous score", f"{ml_assessment['continuous_score']}%"],
            ["Training rows", f"{ml_assessment['dataset_rows']:,}"],
            ["Validation accuracy", f"{ml_assessment['validation_accuracy_pct']}%"],
        ]
        ml_table = Table(ml_data, colWidths=[5*cm, 12*cm])
        ml_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ddd')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafa')]),
            ('TEXTCOLOR', (1,1), (1,1), colors.HexColor(ml_color)),
            ('TEXTCOLOR', (1,2), (1,3), colors.HexColor(ml_color)),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph(ml_assessment["mapping_note"], body_style))
        if ml_assessment["top_signals"]:
            signal_text = " | ".join(
                f"{signal['label']} ({signal['score']}/5)"
                for signal in ml_assessment["top_signals"]
            )
            story.append(Paragraph(f"Strongest mapped signals: {signal_text}", body_style))
        story.append(Spacer(1, 0.3*cm))

    # ── Charts
    story.append(Paragraph("Visual Analysis", heading_style))

    gauge_b64 = make_gauge_chart(scores['composite'], scores['level'], lc)
    radar_b64 = make_radar_chart(scores['psychological_pct'],
                                  scores['behavioural_pct'],
                                  scores['cognitive_pct'])

    gauge_data = base64.b64decode(gauge_b64)
    radar_data = base64.b64decode(radar_b64)

    gauge_buf = io.BytesIO(gauge_data)
    radar_buf = io.BytesIO(radar_data)

    g_img = RLImage(gauge_buf, width=8*cm, height=4.5*cm)
    r_img = RLImage(radar_buf, width=7*cm, height=7*cm)

    chart_table = Table([[g_img, r_img]], colWidths=[9*cm, 8*cm])
    chart_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(chart_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Question Responses
    story.append(Paragraph("Detailed Response Log", heading_style))

    q_data = [["#", "Category", "Scale", "Question (truncated)", "Score"]]
    cat_labels = {'P': 'Psychological', 'B': 'Behavioural', 'C': 'Cognitive'}
    for q in QUESTIONS:
        qid = str(q['id'])
        ans = int(answers.get(qid, 0))
        q_data.append([
            str(q['id']),
            cat_labels[q['category']],
            q['scale'],
            Paragraph(q['text'][:80] + ('…' if len(q['text']) > 80 else ''), body_style),
            q['options'][ans],
        ])

    qt = Table(q_data, colWidths=[0.8*cm, 3.2*cm, 1.8*cm, 9.5*cm, 2*cm])
    qt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#ddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f8f8')]),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (-1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#333')),
    ]))
    story.append(qt)
    story.append(Spacer(1, 0.5*cm))

    # ── Suggestions
    story.append(Paragraph("Personalised Recommendations", heading_style))
    suggestions = get_suggestions(scores)
    for s in suggestions:
        story.append(Paragraph(s, bullet_style))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.5*cm))

    # ── Disclaimer
    disc_style = ParagraphStyle('Disc', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#888'),
        leading=12, alignment=TA_CENTER, borderPad=8)
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ddd')))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠️  This report is generated by an automated screening tool and is NOT a clinical diagnosis. "
        "It is intended for informational and self-awareness purposes only. If you are experiencing significant distress, "
        "please consult a qualified mental health professional. Scales referenced: PSS-10 (Cohen et al.), GAD-7 (Spitzer et al.), "
        "PHQ-9 (Kroenke et al.), DASS-21 (Lovibond), ISI (Morin et al.), CDS (Cognitive Distress Scale).",
        disc_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _interpret(score: float) -> str:
    if score < 20:   return "Minimal"
    elif score < 40: return "Mild"
    elif score < 60: return "Moderate"
    elif score < 80: return "High"
    else:            return "Severe"


# ─────────────────────────────────────────────────────────────
#  HTML PAGE
# ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>MindScan Pro – Stress Detection</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
  :root{
    --bg:#0d1117; --surface:#161b22; --border:#21262d;
    --accent:#7c3aed; --accent2:#06b6d4;
    --green:#2ecc71; --yellow:#f1c40f; --orange:#e67e22;
    --red:#e74c3c; --purple:#8e44ad;
    --text:#e6edf3; --muted:#8b949e;
    --radius:14px; --font-sans:'DM Sans',sans-serif; --font-serif:'DM Serif Display',serif;
    --font-mono:'JetBrains Mono',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font-sans);min-height:100vh;overflow-x:hidden}

  /* ── LAYOUT ── */
  .container{max-width:780px;margin:0 auto;padding:2rem 1.5rem}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:2rem;margin-bottom:1.5rem}

  /* ── HEADER ── */
  header{text-align:center;padding:3rem 1.5rem 1rem}
  header h1{font-family:var(--font-serif);font-size:2.8rem;background:linear-gradient(135deg,#7c3aed,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px}
  header p{color:var(--muted);font-size:1rem;margin-top:.5rem}
  .badge{display:inline-flex;align-items:center;gap:.4rem;background:#7c3aed22;border:1px solid #7c3aed55;color:#a78bfa;border-radius:99px;padding:.3rem .9rem;font-size:.8rem;font-weight:500;margin:.5rem .25rem}

  /* ── INTRO CARD ── */
  .intro-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0}
  .intro-item{background:#0d1117;border-radius:10px;padding:1rem;text-align:center;border:1px solid var(--border)}
  .intro-item .num{font-family:var(--font-mono);font-size:1.6rem;font-weight:700;color:#7c3aed}
  .intro-item span{font-size:.8rem;color:var(--muted);display:block;margin-top:.2rem}
  #userName{width:100%;background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;color:var(--text);font-size:1rem;margin-top:.75rem;font-family:var(--font-sans)}
  #userName:focus{outline:none;border-color:#7c3aed;box-shadow:0 0 0 3px #7c3aed22}

  /* ── BUTTONS ── */
  .btn{display:inline-flex;align-items:center;gap:.5rem;border:none;border-radius:99px;padding:.8rem 2rem;font-family:var(--font-sans);font-size:.95rem;font-weight:600;cursor:pointer;transition:all .2s}
  .btn-primary{background:linear-gradient(135deg,#7c3aed,#4f46e5);color:white}
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 25px #7c3aed44}
  .btn-secondary{background:transparent;border:1.5px solid var(--border);color:var(--muted)}
  .btn-secondary:hover{border-color:#7c3aed;color:#a78bfa}
  .btn-voice{background:linear-gradient(135deg,#e74c3c,#c0392b);color:white;font-size:.85rem;padding:.6rem 1.2rem}
  .btn-voice.listening{animation:pulse-red 1s infinite}
  @keyframes pulse-red{0%,100%{box-shadow:0 0 0 0 #e74c3c66}50%{box-shadow:0 0 0 12px #e74c3c00}}

  /* ── PROGRESS ── */
  .progress-bar{background:var(--border);border-radius:99px;height:6px;margin-bottom:1.5rem;overflow:hidden}
  .progress-fill{background:linear-gradient(90deg,#7c3aed,#06b6d4);height:100%;border-radius:99px;transition:width .4s}
  .progress-label{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-bottom:.5rem}

  /* ── QUESTION CARD ── */
  .q-meta{display:flex;gap:.5rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap}
  .q-cat{font-size:.75rem;font-weight:600;border-radius:99px;padding:.25rem .75rem}
  .cat-P{background:#7c3aed22;color:#a78bfa;border:1px solid #7c3aed44}
  .cat-B{background:#e6782222;color:#fca047;border:1px solid #e6782244}
  .cat-C{background:#06b6d422;color:#22d3ee;border:1px solid #06b6d444}
  .q-scale{font-size:.75rem;color:var(--muted);background:#0d1117;border:1px solid var(--border);border-radius:99px;padding:.2rem .6rem}
  .q-weight{font-size:.75rem;font-family:var(--font-mono);color:#7c3aed}
  .q-text{font-size:1.15rem;color:var(--text);line-height:1.6;margin-bottom:1.5rem}

  /* ── OPTION BUTTONS ── */
  .options{display:flex;flex-direction:column;gap:.6rem}
  .opt{background:#0d1117;border:1.5px solid var(--border);border-radius:10px;padding:.85rem 1.2rem;
       display:flex;align-items:center;gap:1rem;cursor:pointer;transition:all .2s;text-align:left}
  .opt:hover{border-color:#7c3aed;background:#7c3aed11}
  .opt.selected{border-color:#7c3aed;background:#7c3aed22;color:#c4b5fd}
  .opt-num{font-family:var(--font-mono);font-size:.8rem;color:var(--muted);width:1.5rem;flex-shrink:0}
  .opt-label{font-size:.95rem}
  .opt-bar{height:4px;border-radius:99px;background:#7c3aed;margin-top:.4rem;transition:width .3s}

  /* ── VOICE AREA ── */
  .voice-area{background:#0d1117;border:1px dashed var(--border);border-radius:10px;padding:1rem;margin-top:1rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
  .voice-transcript{font-size:.85rem;color:var(--muted);font-style:italic;flex:1;min-width:150px}
  #voiceStatus{font-size:.75rem;color:var(--muted);margin-top:.5rem}
  .wave{display:flex;gap:3px;align-items:center;height:20px}
  .wave span{display:block;width:3px;background:#e74c3c;border-radius:2px;animation:wave .8s ease-in-out infinite}
  .wave span:nth-child(2){animation-delay:.1s}.wave span:nth-child(3){animation-delay:.2s}
  .wave span:nth-child(4){animation-delay:.3s}.wave span:nth-child(5){animation-delay:.4s}
  @keyframes wave{0%,100%{height:4px}50%{height:18px}}
  .wave.hidden{display:none}

  /* ── REPORT ── */
  #report{display:none}
  .score-hero{text-align:center;padding:2rem}
  .score-num{font-family:var(--font-mono);font-size:4rem;font-weight:700;line-height:1}
  .score-label{font-size:1.2rem;font-weight:600;margin-top:.5rem}
  .domain-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0}
  .domain-card{background:#0d1117;border-radius:10px;padding:1.2rem;text-align:center;border:1px solid var(--border)}
  .domain-val{font-family:var(--font-mono);font-size:1.6rem;font-weight:700}
  .domain-name{font-size:.8rem;color:var(--muted);margin-top:.3rem}
  .domain-weight{font-size:.7rem;color:#7c3aed;margin-top:.1rem}
  .suggestions-list li{list-style:none;background:#0d1117;border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;border-left:3px solid #7c3aed;font-size:.95rem;line-height:1.5}
  img.chart{width:100%;border-radius:10px;margin-bottom:1rem}
  .chart-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  .disclaimer{font-size:.8rem;color:var(--muted);text-align:center;line-height:1.6;border-top:1px solid var(--border);padding-top:1rem;margin-top:1rem}
  .download-btn{display:inline-flex;align-items:center;gap:.5rem;background:linear-gradient(135deg,#059669,#10b981);color:white;border:none;border-radius:99px;padding:.9rem 2rem;font-size:1rem;font-weight:600;cursor:pointer;transition:all .2s;text-decoration:none}
  .download-btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px #10b98144}

  /* ── MISC ── */
  .hidden{display:none!important}
  .nav{display:flex;justify-content:space-between;align-items:center;margin-top:1.5rem;flex-wrap:wrap;gap:.75rem}
  .step-dots{display:flex;gap:.4rem;justify-content:center;margin-bottom:1rem}
  .dot{width:6px;height:6px;border-radius:50%;background:var(--border);transition:all .3s}
  .dot.done{background:#7c3aed}
  .dot.active{background:#06b6d4;width:18px;border-radius:3px}
  @media(max-width:520px){
    .intro-grid,.domain-cards,.chart-row{grid-template-columns:1fr!important}
    header h1{font-size:2rem}
  }
</style>
</head>
<body>

<header>
  <h1>MindScan Pro</h1>
  <p>Evidence-based stress detection powered by clinical psychometric scales</p>
  <div style="margin-top:.75rem">
    <span class="badge">🎙️ Voice Enabled</span>
    <span class="badge">📊 PSS-10</span>
    <span class="badge">📊 GAD-7</span>
    <span class="badge">📊 PHQ-9</span>
    <span class="badge">📊 DASS-21</span>
  </div>
  <div style="margin-top:1rem">
    <a href="/dashboard" class="btn btn-secondary" style="text-decoration:none">View Past Responses</a>
  </div>
</header>

<div class="container">

  <!-- ═══ INTRO SCREEN ═══ -->
  <div id="introScreen">
    <div class="card">
      <h2 style="font-family:var(--font-serif);font-size:1.4rem;margin-bottom:.75rem">About This Assessment</h2>
      <p style="color:var(--muted);line-height:1.7;font-size:.95rem">
        MindScan Pro uses a trained questionnaire model to estimate stress from your answers across
        psychological, behavioural, and cognitive dimensions. The final stress score is now generated by
        the integrated ML dataset, while the domain breakdown remains as supporting context.
      </p>
      <div class="intro-grid" style="margin-top:1.5rem">
        <div class="intro-item"><div class="num">18</div><span>Questions</span></div>
        <div class="intro-item"><div class="num">~5 min</div><span>Duration</span></div>
        <div class="intro-item"><div class="num">6</div><span>Clinical Scales</span></div>
      </div>
      <div style="margin-top:1.5rem">
        <div class="intro-grid">
          <div class="intro-item"><div class="num" style="color:#a78bfa">100k</div><span>Training Rows</span></div>
          <div class="intro-item"><div class="num" style="color:#fca047">18</div><span>Mapped ML Features</span></div>
          <div class="intro-item"><div class="num" style="color:#22d3ee">ML</div><span>Primary Scoring</span></div>
        </div>
      </div>
      <input id="userName" type="text" placeholder="Enter your name (optional)" maxlength="50"/>
      <div style="margin-top:1.5rem;text-align:center">
        <button class="btn btn-primary" onclick="startAssessment()">
          Begin Assessment →
        </button>
      </div>
    </div>
  </div>

  <!-- ═══ QUESTION SCREEN ═══ -->
  <div id="questionScreen" class="hidden">
    <div class="progress-label">
      <span id="qProgress">Question 1 of 20</span>
      <span id="qPct">0%</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    <div class="step-dots" id="stepDots"></div>

    <div class="card">
      <div class="q-meta">
        <span class="q-cat" id="qCat">Psychological</span>
        <span class="q-scale" id="qScale">PSS-10</span>
        <span class="q-weight" id="qWeight">ML Input</span>
      </div>
      <p class="q-text" id="qText">Loading...</p>
      <div class="options" id="optionsContainer"></div>

      <!-- Voice -->
      <div class="voice-area" style="margin-top:1.2rem">
        <button class="btn btn-voice" id="voiceBtn" onclick="toggleVoice()">🎙️ Speak Answer</button>
        <div>
          <div class="wave hidden" id="waveAnim"><span></span><span></span><span></span><span></span><span></span></div>
          <div class="voice-transcript" id="voiceTranscript">Press the mic to answer by voice</div>
          <div id="voiceStatus"></div>
        </div>
      </div>
    </div>

    <div class="nav">
      <button class="btn btn-secondary" id="prevBtn" onclick="prevQuestion()" disabled>← Back</button>
      <button class="btn btn-primary" id="nextBtn" onclick="nextQuestion()" disabled>Next →</button>
    </div>
  </div>

  <!-- ═══ REPORT SCREEN ═══ -->
  <div id="report">

    <div class="card" style="text-align:center;padding:2.5rem">
      <div style="font-size:3rem;margin-bottom:1rem" id="reportEmoji">🟢</div>
      <div class="score-num" id="reportScore" style="color:#2ecc71">0%</div>
      <div class="score-label" id="reportLevel">Minimal Stress</div>
      <p style="color:var(--muted);margin-top:.5rem;font-size:.9rem">Primary ML Stress Index</p>
    </div>

    <div class="card" id="mlInsightCard" style="display:none">
      <h3 style="font-family:var(--font-serif);font-size:1.2rem;margin-bottom:1rem">ML Model Summary</h3>
      <p id="mlInsightSummary" style="color:var(--muted);line-height:1.7;font-size:.95rem;margin-bottom:1rem"></p>
      <div class="domain-cards">
        <div class="domain-card">
          <div class="domain-val" id="mlLevel" style="color:#2ecc71">Low</div>
          <div class="domain-name">ML Label</div>
          <div class="domain-weight">18 mapped features</div>
        </div>
        <div class="domain-card">
          <div class="domain-val" id="mlConfidence" style="color:#2ecc71">0%</div>
          <div class="domain-name">Confidence</div>
          <div class="domain-weight">Primary scoring model</div>
        </div>
        <div class="domain-card">
          <div class="domain-val" id="mlRows" style="color:#2ecc71">0</div>
          <div class="domain-name">Training Rows</div>
          <div class="domain-weight">Training dataset</div>
        </div>
      </div>
      <div id="mlTopSignals" style="margin-top:1rem;color:var(--muted);font-size:.92rem;line-height:1.7"></div>
    </div>

    <div class="card">
      <h3 style="font-family:var(--font-serif);font-size:1.2rem;margin-bottom:1rem">Domain Breakdown</h3>
      <div class="domain-cards">
        <div class="domain-card">
          <div class="domain-val" id="pScore" style="color:#a78bfa">0%</div>
          <div class="domain-name">Psychological</div>
          <div class="domain-weight">Supporting breakdown</div>
        </div>
        <div class="domain-card">
          <div class="domain-val" id="bScore" style="color:#fca047">0%</div>
          <div class="domain-name">Behavioural</div>
          <div class="domain-weight">Supporting breakdown</div>
        </div>
        <div class="domain-card">
          <div class="domain-val" id="cScore" style="color:#22d3ee">0%</div>
          <div class="domain-name">Cognitive</div>
          <div class="domain-weight">Supporting breakdown</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3 style="font-family:var(--font-serif);font-size:1.2rem;margin-bottom:1rem">Visual Analysis</h3>
      <div class="chart-row">
        <div><img id="gaugeChart" class="chart" alt="Gauge Chart"/></div>
        <div><img id="radarChart" class="chart" alt="Radar Chart"/></div>
      </div>
      <img id="barChart" class="chart" alt="Bar Chart"/>
    </div>

    <div class="card">
      <h3 style="font-family:var(--font-serif);font-size:1.2rem;margin-bottom:1rem">🧩 Personalised Recommendations</h3>
      <ul class="suggestions-list" id="suggestionsList"></ul>
    </div>

    <div class="card">
      <h3 style="font-family:var(--font-serif);font-size:1.2rem;margin-bottom:1rem">📋 Your Responses</h3>
      <div id="responseLog" style="font-size:.85rem"></div>
    </div>

    <div class="card" style="text-align:center">
      <h3 style="font-family:var(--font-serif);font-size:1.2rem;margin-bottom:.75rem">Download Full Report</h3>
      <p style="color:var(--muted);font-size:.9rem;margin-bottom:1.5rem">Get a detailed PDF report with charts, scores, and recommendations.</p>
      <a id="downloadBtn" class="download-btn" href="#" onclick="downloadPDF(event)">
        📄 Download PDF Report
      </a>
      <div style="margin-top:1.5rem">
        <button class="btn btn-secondary" onclick="restartAssessment()">↩ Retake Assessment</button>
      </div>
    </div>

    <p class="disclaimer">
      ⚠️ This is an automated screening tool, not a clinical diagnosis.<br>
      Scales: PSS-10 · GAD-7 · PHQ-9 · DASS-21 · ISI · Cognitive Distress Scale.<br>
      If you're in distress, please contact a mental health professional.<br>
      India Helplines: iCall 9152987821 · Vandrevala Foundation 1860-2662-345
    </p>
  </div>

</div><!-- /container -->

<script>
// ── DATA ──────────────────────────────────────────────────────
const QUESTIONS = __QUESTIONS_JSON__;
const catLabel = {P:'Psychological', B:'Behavioural', C:'Cognitive'};
const catClass  = {P:'cat-P', B:'cat-B', C:'cat-C'};

let currentQ   = 0;
let answers    = {};   // {id: score}
let userName   = '';
let recognition = null;
let isListening = false;
let speechSynthActive = false;
let utteranceQueue = [];

// ── INIT DOTS ─────────────────────────────────────────────────
function buildDots(){
  const c = document.getElementById('stepDots');
  c.innerHTML = '';
  QUESTIONS.forEach((q,i)=>{
    const d = document.createElement('div');
    d.className = 'dot' + (i===currentQ?' active':(answers[q.id]!==undefined?' done':''));
    c.appendChild(d);
  });
}

// ── START ─────────────────────────────────────────────────────
function startAssessment(){
  userName = document.getElementById('userName').value.trim() || 'User';
  document.getElementById('introScreen').classList.add('hidden');
  document.getElementById('questionScreen').classList.remove('hidden');
  initSpeechRecognition();
  renderQuestion();
}

// ── RENDER QUESTION ───────────────────────────────────────────
function renderQuestion(){
  const q = QUESTIONS[currentQ];
  const total = QUESTIONS.length;

  document.getElementById('qProgress').textContent = `Question ${currentQ+1} of ${total}`;
  const pct = Math.round((currentQ/total)*100);
  document.getElementById('qPct').textContent = pct+'%';
  document.getElementById('progressFill').style.width = pct+'%';

  document.getElementById('qCat').textContent  = catLabel[q.category];
  document.getElementById('qCat').className    = 'q-cat '+catClass[q.category];
  document.getElementById('qScale').textContent = q.scale;
  document.getElementById('qWeight').textContent = 'ML Input';
  document.getElementById('qText').textContent  = q.text;

  const container = document.getElementById('optionsContainer');
  container.innerHTML = '';
  q.options.forEach((opt,i)=>{
    const btn = document.createElement('div');
    btn.className = 'opt' + (answers[q.id]===i?' selected':'');
    btn.innerHTML = `<span class="opt-num">${i}</span><span class="opt-label">${opt}</span>`;
    btn.onclick = ()=>selectAnswer(i);
    container.appendChild(btn);
  });

  document.getElementById('prevBtn').disabled = currentQ === 0;
  document.getElementById('nextBtn').disabled = answers[q.id] === undefined;
  document.getElementById('nextBtn').textContent = currentQ === total-1 ? 'Finish ✓' : 'Next →';

  document.getElementById('voiceTranscript').textContent = 'Press the mic to answer by voice';
  document.getElementById('voiceStatus').textContent = '';

  buildDots();

  // TTS – read question aloud
  speakText(q.text + '. Options: ' + q.options.map((o,i)=>i+': '+o).join(', '));
}

// ── SELECT ANSWER ─────────────────────────────────────────────
function selectAnswer(score){
  const q = QUESTIONS[currentQ];
  answers[q.id] = score;
  // Highlight
  document.querySelectorAll('.opt').forEach((el,i)=>{
    el.classList.toggle('selected', i===score);
  });
  document.getElementById('nextBtn').disabled = false;
  buildDots();
}

// ── NAV ───────────────────────────────────────────────────────
function nextQuestion(){
  if(answers[QUESTIONS[currentQ].id] === undefined) return;
  if(currentQ < QUESTIONS.length-1){
    currentQ++;
    renderQuestion();
  } else {
    submitAnswers();
  }
}

function prevQuestion(){
  if(currentQ > 0){ currentQ--; renderQuestion(); }
}

// ── SUBMIT ────────────────────────────────────────────────────
async function submitAnswers(){
  stopVoice();
  document.getElementById('questionScreen').classList.add('hidden');
  document.getElementById('report').style.display = 'block';
  document.getElementById('report').scrollIntoView({behavior:'smooth'});
  showReportLoading();

  const payload = {answers: Object.fromEntries(
    Object.entries(answers).map(([k,v])=>[k,String(v)])
  ), name: userName};

  try{
    const res  = await fetch('/score', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await res.json();
    if(!res.ok){
      throw new Error(data.error || 'Questionnaire ML scoring is unavailable.');
    }
    populateReport(data, payload.answers);
  } catch(e){
    console.error(e);
    showReportError(e.message || 'Questionnaire ML scoring is unavailable.');
  }
}

function setChartsVisible(visible){
  ['gaugeChart', 'radarChart', 'barChart'].forEach((id) => {
    const el = document.getElementById(id);
    if(el){
      el.style.display = visible ? 'block' : 'none';
      if(!visible){
        el.removeAttribute('src');
      }
    }
  });
}

function showReportLoading(){
  document.getElementById('reportEmoji').textContent  = '...';
  document.getElementById('reportScore').textContent  = 'Loading';
  document.getElementById('reportScore').style.color  = '#f1c40f';
  document.getElementById('reportLevel').textContent  = 'Preparing questionnaire ML result...';
  document.getElementById('reportLevel').style.color  = '#f1c40f';

  document.getElementById('pScore').textContent = '--';
  document.getElementById('bScore').textContent = '--';
  document.getElementById('cScore').textContent = '--';

  setChartsVisible(false);

  document.getElementById('mlInsightCard').style.display = 'none';
  document.getElementById('mlTopSignals').innerHTML = '';
  document.getElementById('suggestionsList').innerHTML =
    '<li>Building your ML-backed stress result. This can take a little longer on the first run.</li>';
  document.getElementById('responseLog').innerHTML =
    '<div style="color:#8b949e">Waiting for questionnaire ML scoring...</div>';
  document.getElementById('downloadBtn').style.display = 'none';
}

function showReportError(message){
  document.getElementById('reportEmoji').textContent  = '⚠';
  document.getElementById('reportScore').textContent  = 'ML Required';
  document.getElementById('reportScore').style.color  = '#e74c3c';
  document.getElementById('reportLevel').textContent  = message;
  document.getElementById('reportLevel').style.color  = '#e74c3c';

  document.getElementById('pScore').textContent = '--';
  document.getElementById('bScore').textContent = '--';
  document.getElementById('cScore').textContent = '--';

  setChartsVisible(false);

  document.getElementById('mlInsightCard').style.display = 'none';
  document.getElementById('mlTopSignals').innerHTML = '';
  document.getElementById('suggestionsList').innerHTML =
    '<li>The assessment result is only shown when the questionnaire ML model is available.</li>';
  document.getElementById('responseLog').innerHTML =
    '<div style="color:#8b949e">No ML-backed result was generated for this submission.</div>';
  document.getElementById('downloadBtn').style.display = 'none';
}

// ── POPULATE REPORT ───────────────────────────────────────────
function populateReport(data, rawAnswers){
  const levelColors = {Low:'#2ecc71',Minimal:'#2ecc71',Mild:'#f1c40f',Moderate:'#f1c40f',High:'#e67e22',Severe:'#e74c3c'};
  const mlLevelColors = {Low:'#2ecc71',Moderate:'#f1c40f',High:'#e67e22',Severe:'#e74c3c'};
  const lc = levelColors[data.level] || '#7c3aed';

  document.getElementById('reportEmoji').textContent  = data.emoji;
  document.getElementById('reportScore').textContent  = data.composite + '%';
  document.getElementById('reportScore').style.color  = lc;
  document.getElementById('reportLevel').textContent  = data.level + ' Stress';
  document.getElementById('reportLevel').style.color  = lc;

  document.getElementById('pScore').textContent = data.psychological_pct + '%';
  document.getElementById('bScore').textContent = data.behavioural_pct   + '%';
  document.getElementById('cScore').textContent = data.cognitive_pct     + '%';

  setChartsVisible(true);
  document.getElementById('gaugeChart').src = 'data:image/png;base64,' + data.gauge_b64;
  document.getElementById('radarChart').src = 'data:image/png;base64,' + data.radar_b64;
  document.getElementById('barChart').src   = 'data:image/png;base64,' + data.bar_b64;
  document.getElementById('downloadBtn').style.display = 'inline-flex';

  const mlCard = document.getElementById('mlInsightCard');
  if(data.ml_assessment){
    const ml = data.ml_assessment;
    const mlColor = mlLevelColors[ml.stress_label] || '#7c3aed';
    mlCard.style.display = 'block';
    document.getElementById('mlLevel').textContent = ml.stress_label;
    document.getElementById('mlLevel').style.color = mlColor;
    document.getElementById('mlConfidence').textContent = ml.confidence_pct + '%';
    document.getElementById('mlConfidence').style.color = mlColor;
    document.getElementById('mlRows').textContent = (ml.dataset_rows || 0).toLocaleString();
    document.getElementById('mlRows').style.color = mlColor;
    document.getElementById('mlInsightSummary').textContent =
      'Primary model result: ' + ml.stress_label +
      ' pattern with ' + ml.confidence_pct + '% confidence. ' +
      'The questionnaire model was trained on ' + (ml.dataset_rows || 0).toLocaleString() +
      ' questionnaire rows and validated at ' + ml.validation_accuracy_pct + '%.';
    document.getElementById('mlTopSignals').innerHTML = (ml.top_signals || []).length
      ? '<strong style="color:#e6edf3">Strongest mapped signals:</strong> ' +
        ml.top_signals.map(signal => `${signal.label} (${signal.score}/5)`).join(' · ')
      : '';
  } else {
    mlCard.style.display = 'none';
  }

  const sl = document.getElementById('suggestionsList');
  sl.innerHTML = '';
  (data.suggestions||[]).forEach(s=>{
    const li = document.createElement('li');
    li.textContent = s;
    sl.appendChild(li);
  });

  // Response log
  const catLabel = {P:'Psychological', B:'Behavioural', C:'Cognitive'};
  const rl = document.getElementById('responseLog');
  rl.innerHTML = '<table style="width:100%;border-collapse:collapse">' +
    '<tr style="background:#0d1117;color:#8b949e;font-size:.8rem">' +
    '<th style="padding:6px;text-align:left">#</th>' +
    '<th style="padding:6px;text-align:left">Domain</th>' +
    '<th style="padding:6px;text-align:left">Scale</th>' +
    '<th style="padding:6px;text-align:left">Question</th>' +
    '<th style="padding:6px;text-align:center">Answer</th>' +
    '</tr>' +
    QUESTIONS.map((q,i)=>{
      const ans = rawAnswers[q.id] !== undefined ? q.options[rawAnswers[q.id]] : '—';
      const bg = i%2===0?'#0d1117':'#161b22';
      return `<tr style="background:${bg};font-size:.82rem">
        <td style="padding:5px 6px;color:#555">${q.id}</td>
        <td style="padding:5px 6px;color:#8b949e">${catLabel[q.category]}</td>
        <td style="padding:5px 6px;color:#7c3aed">${q.scale}</td>
        <td style="padding:5px 6px;color:#e6edf3">${q.text.slice(0,70)}${q.text.length>70?'…':''}</td>
        <td style="padding:5px 6px;text-align:center;color:#06b6d4;font-weight:600">${ans}</td>
      </tr>`;
    }).join('') + '</table>';

  speakText('Assessment complete. The ML model estimates your stress level as ' + data.level + ' at ' + data.composite + ' percent. Please review your detailed report and recommendations.');
}

// ── DOWNLOAD PDF ──────────────────────────────────────────────
async function downloadPDF(e){
  e.preventDefault();
  const payload = {
    answers: Object.fromEntries(Object.entries(answers).map(([k,v])=>[k,String(v)])),
    name: userName
  };
  try{
    const res  = await fetch('/pdf', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!res.ok){
      throw new Error(await res.text() || 'Unable to generate PDF without questionnaire ML scoring.');
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'MindScan_Stress_Report.pdf';
    a.click();
  } catch(e){
    console.error(e);
    showReportError(e.message || 'Unable to generate PDF without questionnaire ML scoring.');
  }
}

// ── RESTART ───────────────────────────────────────────────────
function restartAssessment(){
  currentQ = 0; answers = {};
  document.getElementById('report').style.display = 'none';
  document.getElementById('introScreen').classList.remove('hidden');
  window.scrollTo({top:0,behavior:'smooth'});
}

// ── TTS ───────────────────────────────────────────────────────
function speakText(text){
  if(!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.92; u.pitch = 1; u.lang = 'en-IN';
  // Prefer Indian English voice if available
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v=>v.lang.includes('en-IN')) ||
                    voices.find(v=>v.lang.includes('en-GB')) ||
                    voices.find(v=>v.lang.startsWith('en'));
  if(preferred) u.voice = preferred;
  window.speechSynthesis.speak(u);
}

// ── STT ───────────────────────────────────────────────────────
function initSpeechRecognition(){
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SpeechRec){ document.getElementById('voiceBtn').disabled = true; return; }

  recognition = new SpeechRec();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-IN';

  recognition.onresult = (event)=>{
    let transcript = '';
    for(let i=event.resultIndex;i<event.results.length;i++){
      transcript += event.results[i][0].transcript;
    }
    document.getElementById('voiceTranscript').textContent = '🎤 ' + transcript;
    if(event.results[event.resultIndex].isFinal){
      processVoiceInput(transcript.toLowerCase().trim());
    }
  };

  recognition.onend  = ()=>{ stopVoice(); };
  recognition.onerror = (e)=>{
    document.getElementById('voiceStatus').textContent = 'Voice error: ' + e.error;
    stopVoice();
  };
}

function toggleVoice(){
  if(isListening){ stopVoice(); return; }
  if(!recognition){ return; }
  window.speechSynthesis.cancel(); // Stop TTS before listening
  recognition.start();
  isListening = true;
  document.getElementById('voiceBtn').textContent = '⏹ Stop Listening';
  document.getElementById('voiceBtn').classList.add('listening');
  document.getElementById('waveAnim').classList.remove('hidden');
  document.getElementById('voiceStatus').textContent = 'Listening… say a number (0-4) or option name';
}

function stopVoice(){
  if(recognition && isListening){ try{ recognition.stop(); }catch(e){} }
  isListening = false;
  document.getElementById('voiceBtn').textContent = '🎙️ Speak Answer';
  document.getElementById('voiceBtn').classList.remove('listening');
  document.getElementById('waveAnim').classList.add('hidden');
}

function processVoiceInput(text){
  const q = QUESTIONS[currentQ];
  // Match by number
  const numMatch = text.match(/\b([0-4])\b/);
  if(numMatch){ selectAnswer(parseInt(numMatch[1])); return; }

  // Match by keyword
  const kw = {
    never:0, 'not at all':0, zero:0,
    rarely:1, seldom:1, little:1, one:1,
    sometimes:2, occasionally:2, two:2, often:2,
    frequently:3, three:3, most:3,
    always:4, every:4, four:4, daily:4, constantly:4,
  };
  for(const [word, score] of Object.entries(kw)){
    if(text.includes(word)){ selectAnswer(score); return; }
  }
  document.getElementById('voiceStatus').textContent = '⚠️ Not recognised. Say 0–4 or Never/Rarely/Sometimes/Often/Always';
}

// Preload voices
if('speechSynthesis' in window){
  speechSynthesis.onvoiceschanged = ()=>{ speechSynthesis.getVoices(); };
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────
#  HTTP SERVER
# ─────────────────────────────────────────────────────────────

class StressHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == '/':
            questions_json = json.dumps(QUESTIONS)
            page = HTML_PAGE.replace('__QUESTIONS_JSON__', questions_json)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page.encode('utf-8'))
        elif self.path == '/dashboard':
            data = get_all_responses()
            html = '<html><head><title>Dashboard</title><style>body{font-family:sans-serif;background:#0d1117;color:white;padding:2rem} table{width:100%;border-collapse:collapse;margin-top:20px} th,td{border:1px solid #333;padding:10px;text-align:left} a{color:#7c3aed;text-decoration:none} a:hover{text-decoration:underline}</style></head><body>'
            html += '<h2>Stored Responses Dashboard</h2>'
            html += '<a href="/">← Back to Assessment</a><br>'
            html += '<table><tr><th>ID</th><th>Name</th><th>Level</th><th>Score</th><th>Date</th></tr>'
            for r in data:
                html += f'<tr><td>{r["id"]}</td><td>{r["name"]}</td><td>{r["level"]}</td><td>{r["composite"]}%</td><td>{r["created_at"]}</td></tr>'
            html += '</table></body></html>'
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)
        data   = json.loads(body)

        if self.path == '/score':
            try:
                scores  = compute_scores(data.get('answers', {}))
                name    = data.get('name', 'Anonymous')
                save_to_db(name, data.get('answers', {}), scores)
                ml_assessment = scores.get('ml_assessment')
                sugg    = get_suggestions(scores)
                gauge   = make_gauge_chart(scores['composite'], scores['level'], scores['color'])
                radar   = make_radar_chart(scores['psychological_pct'],
                                           scores['behavioural_pct'],
                                           scores['cognitive_pct'])
                bar     = make_bar_chart(scores['psychological_pct'],
                                         scores['behavioural_pct'],
                                         scores['cognitive_pct'])
                resp = {**scores, 'suggestions': sugg,
                        'gauge_b64': gauge, 'radar_b64': radar, 'bar_b64': bar,
                        'ml_assessment': ml_assessment}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())
            except Exception as exc:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())

        elif self.path == '/pdf':
            try:
                scores  = compute_scores(data.get('answers', {}))
                name    = data.get('name', 'User')
                pdf_bytes = generate_pdf_report(scores, data.get('answers', {}), name)
                self.send_response(200)
                self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Disposition', 'attachment; filename="MindScan_Report.pdf"')
                self.send_header('Content-Length', str(len(pdf_bytes)))
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as exc:
                self.send_response(503)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(str(exc).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

PORT = 5050

def run():
    init_db()
    threading.Thread(target=get_ml_model_bundle, daemon=True).start()
    server = HTTPServer(('127.0.0.1', PORT), StressHandler)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          MindScan Pro – Stress Detection System          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  🌐  Open in browser: http://localhost:{PORT}            ║
║                                                          ║
║  Features:                                               ║
║  ✅  ML-first questionnaire scoring pipeline             ║
║  ✅  Voice input (STT) + Text-to-Speech questions        ║
║  ✅  PSS-10, GAD-7, PHQ-9, DASS-21, ISI, CDS scales      ║
║  ✅  Gauge, Radar & Bar chart visualisations             ║
║  ✅  Personalised recommendations                        ║
║  ✅  Downloadable PDF report                             ║
║                                                          ║
║  Press Ctrl+C to stop.                                   ║
╚══════════════════════════════════════════════════════════╝
""")
    threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋  MindScan Pro stopped. Goodbye!")


if __name__ == '__main__':
    run()
