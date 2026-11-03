import csv
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "models" / "ml"
DL_DIR = BASE_DIR / "deep_learning" / "models"
FUSION_DIR = DL_DIR / "fusion_runs"
SPLITS_DIR = BASE_DIR / "deployment_headers"

MRI_MODEL_PATH = (
    BASE_DIR / "transfer_learning" / "models"
    / "acdc_resnet50_final.keras"
)
MRI_CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

DATASETS = {
    "heart": {"label": "Heart", "features": 13},
    "metabolic": {"label": "Metabolic", "features": 20},
    "diabetes": {"label": "Diabetes", "features": 21},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

_mri_model = None
_mri_error = None
_ml_models = {}
_dl_models = {}
_fusion_components = {}


def get_feature_names(dataset):
    """Read the original training feature order from the saved CSV."""
    path = SPLITS_DIR / f"{dataset}_X_train.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path.name}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return next(csv.reader(file))


def get_mri_model():
    global _mri_model, _mri_error

    if _mri_model is not None:
        return _mri_model

    if _mri_error:
        raise RuntimeError(_mri_error)

    try:
        if not MRI_MODEL_PATH.is_file():
            raise FileNotFoundError(MRI_MODEL_PATH.name)

        _mri_model = tf.keras.models.load_model(
            MRI_MODEL_PATH, compile=False
        )
        return _mri_model
    except Exception as exc:
        _mri_error = str(exc)
        raise RuntimeError(_mri_error) from exc


def get_ml_model(dataset):
    if dataset in _ml_models:
        return _ml_models[dataset]

    path = ML_DIR / f"{dataset}_XGBoost.pkl"

    if not path.is_file():
        raise FileNotFoundError(
            f"Saved XGBoost model is missing for {dataset}."
        )

    model = joblib.load(path)

    if not callable(getattr(model, "predict", None)):
        raise RuntimeError(f"{path.name} is not a usable classifier.")

    if not callable(getattr(model, "predict_proba", None)):
        raise RuntimeError(
            f"{path.name} does not support probability prediction."
        )

    _ml_models[dataset] = model
    return model


def get_dl_model(dataset):
    if dataset in _dl_models:
        return _dl_models[dataset]

    path = DL_DIR / f"{dataset}_dnn.keras"

    if not path.is_file():
        raise FileNotFoundError(
            f"Saved DNN model is missing for {dataset}."
        )

    model = tf.keras.models.load_model(path, compile=False)
    _dl_models[dataset] = model
    return model


def get_cnn_gru_model(dataset, model_type):
    if model_type not in {"cnn", "gru"}:
        raise ValueError("Unsupported model type.")

    filename = (
        f"{dataset}_cnn1d.keras"
        if model_type == "cnn"
        else f"{dataset}_gru.keras"
    )
    path = DL_DIR / filename
    cache_key = f"{dataset}_{model_type}"

    if cache_key in _dl_models:
        return _dl_models[cache_key]

    if not path.is_file():
        raise FileNotFoundError(
            f"Saved {model_type.upper()} model is missing: {filename}"
        )

    model = tf.keras.models.load_model(path, compile=False)
    _dl_models[cache_key] = model
    return model



def get_fusion_components(dataset):
    if dataset in _fusion_components:
        return _fusion_components[dataset]

    cnn_path = DL_DIR / f"{dataset}_cnn1d.keras"
    gru_path = DL_DIR / f"{dataset}_gru.keras"
    fusion_path = FUSION_DIR / f"{dataset}_cnn_gru_fusion.keras"

    for path in (cnn_path, gru_path, fusion_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required fusion model not found: {path}")

    cnn = tf.keras.models.load_model(cnn_path, compile=False)
    gru = tf.keras.models.load_model(gru_path, compile=False)
    fusion_head = tf.keras.models.load_model(fusion_path, compile=False)

    cnn_features = tf.keras.Model(
        inputs=cnn.inputs,
        outputs=cnn.get_layer("dense").output
    )

    gru_features = tf.keras.Model(
        inputs=gru.inputs,
        outputs=gru.get_layer("gru").output
    )

    cnn_features.trainable = False
    gru_features.trainable = False

    components = (cnn_features, gru_features, fusion_head)
    _fusion_components[dataset] = components

    return components


def get_model_status(dataset, model_type):
    if model_type != "fusion":
        return {
            "available": False,
            "file": None
        }

    cnn_path = DL_DIR / f"{dataset}_cnn1d.keras"
    gru_path = DL_DIR / f"{dataset}_gru.keras"
    fusion_path = FUSION_DIR / f"{dataset}_cnn_gru_fusion.keras"

    available = all(
        path.is_file()
        for path in (cnn_path, gru_path, fusion_path)
    )

    return {
        "available": available,
        "file": fusion_path.name if available else None
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/assets/<path:asset_path>")
def project_asset(asset_path):
    allowed = (
        asset_path.startswith("evaluation/")
        or asset_path.startswith("transfer_learning/results/")
    )

    if not allowed or ".." in Path(asset_path).parts:
        return jsonify({"error": "Asset not found"}), 404

    return send_from_directory(BASE_DIR, asset_path)


@app.get("/api/health")
def health():

    fusion_status = {}

    for dataset in DATASETS:
        status = get_model_status(dataset, "fusion")

        fusion_status[dataset] = {
            "model_file": status["file"],
            "file_exists": status["available"],
            "model_loaded": dataset in _fusion_components
        }

    all_files_exist = all(
        item["file_exists"]
        for item in fusion_status.values()
    )

    any_model_loaded = any(
        item["model_loaded"]
        for item in fusion_status.values()
    )

    return jsonify({
        "backend": "online",

        "model_file": "CNN-GRU Fusion (Heart, Metabolic, Diabetes)",
        "file_exists": all_files_exist,
        "model_loaded": any_model_loaded,

        "fusion_models": fusion_status,

        "mri_model_file": MRI_MODEL_PATH.name,
        "mri_model_file_exists": MRI_MODEL_PATH.is_file(),
        "mri_model_loaded": _mri_model is not None,
        "mri_model_error": _mri_error,
        "mri_classes": MRI_CLASSES
    })


@app.get("/api/models")
def models():
    result = {}

    for dataset, info in DATASETS.items():
        try:
            features = get_feature_names(dataset)
        except (OSError, StopIteration, csv.Error) as exc:
            return jsonify({
                "error": f"Could not read features for {dataset}: {exc}"
            }), 500

        result[dataset] = {
             "label": info["label"],
    "features": features,
    "feature_count": len(features),
    "fusion": get_model_status(dataset, "fusion"),
        }

    return jsonify({
        "datasets": result,
        "notice": "Research demo only; not for medical diagnosis.",
    })


@app.post("/api/predict/tabular")
def predict_tabular():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON request body."}), 400

    dataset = data.get("dataset")
    model_type = data.get("model_type")
    features_data = data.get("features")

    if dataset not in DATASETS:
        return jsonify({"error": "Choose heart, metabolic, or diabetes."}), 400

   

    if model_type != "fusion":
        return jsonify({
            "error": "Choose model_type: fusion."
        }), 400

    if not isinstance(features_data, dict):
        return jsonify({"error": "The features field must be an object."}), 400

    try:
        feature_names = get_feature_names(dataset)

        missing = [name for name in feature_names if name not in features_data]
        if missing:
            return jsonify({
                "error": "Some features are missing.",
                "missing_features": missing,
            }), 400

        values = []
        for name in feature_names:
            value = features_data[name]

            if value is None or isinstance(value, bool):
                return jsonify({
                    "error": f"Enter a numeric value for {name}."
                }), 400

            try:
                number = float(value)
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Enter a numeric value for {name}."
                }), 400

            if not np.isfinite(number):
                return jsonify({
                    "error": f"{name} must be a finite number."
                }), 400

            values.append(number)

        # Preserve the exact feature order used during training.
        # The original training code does not apply StandardScaler.
        batch = np.asarray([values], dtype=np.float32)

        
        cnn_features, gru_features, fusion_head = get_fusion_components(dataset)

        expected_features = DATASETS[dataset]["features"]
        if len(feature_names) != expected_features or len(values) != expected_features:
            raise RuntimeError(
                f"{dataset} expects {expected_features} features, "
                f"but received {len(values)}."
            )

        sequence_batch = batch[..., np.newaxis]

        cnn_output = np.asarray(
            cnn_features.predict(sequence_batch, verbose=0)
        )
        gru_output = np.asarray(
            gru_features.predict(sequence_batch, verbose=0)
        )

        if cnn_output.ndim != 2 or gru_output.ndim != 2:
            raise RuntimeError(
                "CNN and GRU feature extractors must return 2D outputs."
            )

        fused_features = np.concatenate(
            [cnn_output, gru_output], axis=1
        ).astype(np.float32)

        raw_output = np.asarray(
            fusion_head.predict(fused_features, verbose=0)
        ).reshape(-1)

        if raw_output.size != 1 or not np.isfinite(raw_output[0]):
            raise RuntimeError(
                "CNN-GRU Fusion returned an invalid prediction."
            )

        positive_probability = float(
            np.clip(raw_output[0], 0.0, 1.0)
        )
        predicted_label = (
            "1" if positive_probability >= 0.5 else "0"
        )
        model_name = f"{dataset}_cnn_gru_fusion.keras"



        return jsonify({
            "dataset": dataset,
            "model_type": model_type,
            "model": model_name,
            "prediction": predicted_label,
            "positive_class": "1",
            "positive_probability": round(positive_probability * 100, 2),
            "threshold": 0.5,
            "notice": (
                "Research demo only. This prediction is not a diagnosis "
                "and must not guide treatment."
            ),
        })

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Tabular prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the server terminal logs."
        }), 500


@app.post("/api/predict/mri")
def predict_mri():
    if "image" not in request.files:
        return jsonify({
            "error": "Upload an MRI image using the 'image' field."
        }), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "Please choose an image first."}), 400

    if Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Supported formats: PNG, JPG, JPEG, BMP."
        }), 400

    try:
        with Image.open(uploaded.stream) as source:
            image = source.convert("RGB").resize((224, 224))
            array = np.asarray(image, dtype=np.float32)

        # Preserve the existing MRI preprocessing convention (0-255).
        batch = np.expand_dims(array, axis=0)
        model = get_mri_model()
        probabilities = np.asarray(model.predict(batch, verbose=0))[0]

        if len(probabilities) != len(MRI_CLASSES):
            return jsonify({
                "error": "Model output does not match the five classes."
            }), 500

        top_index = int(np.argmax(probabilities))

        return jsonify({
            "prediction": MRI_CLASSES[top_index],
            "probability": round(float(probabilities[top_index]) * 100, 2),
            "probabilities": [
                {
                    "class": name,
                    "probability": round(float(score) * 100, 2),
                }
                for name, score in sorted(
                    zip(MRI_CLASSES, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
            "model": MRI_MODEL_PATH.name,
            "notice": "Research demo only. This is not a medical diagnosis.",
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "The uploaded file is not a valid readable image."
        }), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("MRI prediction failed")
        return jsonify({
            "error": "MRI prediction failed. Check the server logs."
        }), 500



@app.errorhandler(413)
def too_large(_error):
    return jsonify({
        "error": "Upload is too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
