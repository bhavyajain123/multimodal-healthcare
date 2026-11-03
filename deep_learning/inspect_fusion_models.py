from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "deep_learning" / "models"

for dataset in ["heart", "metabolic", "diabetes"]:
    for model_type in ["cnn1d", "gru"]:
        path = MODEL_DIR / f"{dataset}_{model_type}.keras"

        print("\n" + "=" * 60, flush=True)
        print(f"MODEL: {dataset.upper()} - {model_type.upper()}", flush=True)
        print(f"Path: {path}", flush=True)

        if not path.exists():
            print("MODEL FILE NOT FOUND", flush=True)
            continue

        try:
            model = tf.keras.models.load_model(path, compile=False)

            print("INPUT:", model.input_shape, flush=True)
            print("OUTPUT:", model.output_shape, flush=True)
            print("LAYERS:", flush=True)

            for i, layer in enumerate(model.layers):
                try:
                    shape = layer.output.shape
                except Exception:
                    shape = "unknown"

                print(
                    f"{i}: {layer.name} | "
                    f"{layer.__class__.__name__} | {shape}",
                    flush=True
                )

        except Exception as exc:
            print(f"ERROR loading model: {exc}", flush=True)

print("\nInspection finished.", flush=True)
