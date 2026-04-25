from pathlib import Path

import sklearn


_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent
_RUNTIME_MODEL_ROOT = _REPO_ROOT / "data" / "runtime_models"


def current_sklearn_version() -> str:
    return str(sklearn.__version__)


def _version_tag() -> str:
    version = current_sklearn_version().replace(" ", "_")
    return f"sklearn-{version}"


def runtime_model_root() -> Path:
    _RUNTIME_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    return _RUNTIME_MODEL_ROOT


def runtime_model_dir() -> Path:
    return runtime_model_root() / _version_tag()


def ensure_runtime_model_dir() -> Path:
    target = runtime_model_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


def stress_model_path() -> Path:
    return runtime_model_dir() / "stress_model.pkl"


def stress_shap_model_path() -> Path:
    return runtime_model_dir() / "stress_model_shap.pkl"


def stress_meta_path() -> Path:
    return runtime_model_dir() / "stress_model_meta.json"


def questionnaire_pipeline_path() -> Path:
    return runtime_model_dir() / "questionnaire_stress_pipeline.pkl"


def questionnaire_pipeline_meta_path() -> Path:
    return runtime_model_dir() / "questionnaire_stress_pipeline_meta.json"


def shared_stress_model_path() -> Path:
    return runtime_model_root() / "stress_model.pkl"


def shared_stress_shap_model_path() -> Path:
    return runtime_model_root() / "stress_model_shap.pkl"


def shared_stress_meta_path() -> Path:
    return runtime_model_root() / "stress_model_meta.json"


def legacy_stress_model_path() -> Path:
    return _MODULE_DIR / "stress_model.pkl"


def legacy_stress_shap_model_path() -> Path:
    return _MODULE_DIR / "stress_model_shap.pkl"


def legacy_stress_meta_path() -> Path:
    return _MODULE_DIR / "stress_model_meta.json"
