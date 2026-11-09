"""Train an experimental, fully local "ey rocky" wake-word model.

The resulting model accepts exactly one second of mono 16 kHz float PCM and
returns a wake-word probability. See README.md for limitations and usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import av
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE


def decode_audio(path: Path) -> np.ndarray:
    container = av.open(str(path))
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise ValueError(f"No audio stream in {path}")
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for converted in resampler.resample(frame):
            chunks.append(converted.to_ndarray().reshape(-1))
    for converted in resampler.resample(None):
        chunks.append(converted.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        raise ValueError(f"Could not decode audio in {path}")
    audio = np.concatenate(chunks).astype(np.float32)
    return np.nan_to_num(audio, copy=False)


def fit_clip(audio: np.ndarray, offset: int | None = None) -> np.ndarray:
    if len(audio) >= CLIP_SAMPLES:
        if offset is None:
            offset = max(0, (len(audio) - CLIP_SAMPLES) // 2)
        audio = audio[offset : offset + CLIP_SAMPLES]
    else:
        missing = CLIP_SAMPLES - len(audio)
        left = missing // 2
        audio = np.pad(audio, (left, missing - left))
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def augment_positive(audio: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Small time-scale changes preserve the phrase while varying its cadence.
    rate = float(rng.uniform(0.94, 1.06))
    new_length = max(1, int(len(audio) / rate))
    positions = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)
    stretched = np.interp(positions, np.arange(len(audio), dtype=np.float32), audio)
    sample = fit_clip(stretched)
    shift = int(rng.integers(-1200, 1201))
    sample = np.roll(sample, shift)
    if shift > 0:
        sample[:shift] = 0
    elif shift < 0:
        sample[shift:] = 0

    sample *= float(rng.uniform(0.45, 1.25))
    if len(noise) >= CLIP_SAMPLES:
        start = int(rng.integers(0, len(noise) - CLIP_SAMPLES + 1))
        background = noise[start : start + CLIP_SAMPLES].copy()
    else:
        background = fit_clip(noise)
    if rng.random() < 0.8:
        snr_db = float(rng.uniform(5.0, 22.0))
        background *= rms(sample) / (rms(background) * (10.0 ** (snr_db / 20.0)))
        sample += background
    return np.clip(sample, -1.0, 1.0).astype(np.float32)


def make_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(CLIP_SAMPLES, 1), name="pcm_16khz")
    x = inputs
    for filters, kernel, stride in ((16, 101, 4), (24, 9, 4), (32, 9, 4), (48, 7, 2)):
        x = tf.keras.layers.Conv1D(filters, kernel, strides=stride, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if stride == 4:
            x = tf.keras.layers.Dropout(0.12)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="wake_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="ey_rocky_kws")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path(r"C:\Users\jvrsn\OneDrive\Escritorio\Proyectos\audio"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    paths = sorted(p for p in args.audio_dir.iterdir() if p.is_file() and p.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".m4a"})
    if not paths:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    positives: list[np.ndarray] = []
    noise_audio: np.ndarray | None = None
    hashes: set[str] = set()
    durations: list[tuple[str, float]] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            print(f"Skipping byte-identical duplicate: {path.name}")
            continue
        hashes.add(digest)
        audio = decode_audio(path)
        duration = len(audio) / SAMPLE_RATE
        durations.append((path.name, duration))
        if duration > 10:
            if noise_audio is None or len(audio) > len(noise_audio):
                noise_audio = audio
            print(f"Using as background/noise: {path.name} ({duration:.1f}s)")
        else:
            positives.append(fit_clip(audio))

    if noise_audio is None or len(positives) < 8:
        raise SystemExit("Expected short positive clips plus one background recording longer than 10 seconds.")
    assert noise_audio is not None
    order = rng.permutation(len(positives))
    val_count = max(3, round(len(positives) * 0.2))
    val_ids = set(order[:val_count].tolist())
    train_pos = [clip for i, clip in enumerate(positives) if i not in val_ids]
    val_pos = [clip for i, clip in enumerate(positives) if i in val_ids]

    train_x: list[np.ndarray] = []
    train_y: list[float] = []
    for clip in train_pos:
        train_x.append(clip)
        train_y.append(1.0)
        for _ in range(12):
            train_x.append(augment_positive(clip, noise_audio, rng))
            train_y.append(1.0)

    noise_clips = [fit_clip(noise_audio, i) for i in range(0, len(noise_audio) - CLIP_SAMPLES + 1, CLIP_SAMPLES // 2)]
    if not noise_clips:
        noise_clips = [fit_clip(noise_audio)]
    # Re-sample and vary the provided noise file; these are not substitutes for
    # recordings of ordinary speech, and the report explicitly says so.
    while len(train_y) < sum(train_y) * 2:
        raw = noise_clips[int(rng.integers(0, len(noise_clips)))].copy()
        raw *= float(rng.uniform(0.35, 1.0))
        train_x.append(np.clip(raw, -1.0, 1.0).astype(np.float32))
        train_y.append(0.0)
    val_noise = noise_clips[: min(12, len(noise_clips))]
    val_x = val_pos + val_noise
    val_y = [1.0] * len(val_pos) + [0.0] * len(val_noise)

    X = np.asarray(train_x, dtype=np.float32)[..., None]
    y = np.asarray(train_y, dtype=np.float32)
    vx = np.asarray(val_x, dtype=np.float32)[..., None]
    vy = np.asarray(val_y, dtype=np.float32)
    shuffle = rng.permutation(len(y))
    X, y = X[shuffle], y[shuffle]
    print(f"Training on {len(train_pos)} positive source clips, validating on {len(val_pos)} held-out clips.")
    print(f"Prepared {int(y.sum())} positive and {int(len(y)-y.sum())} negative training windows.")
    print("WARNING: validation negatives are only this noise recording; no speech-negative evaluation is available.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    callbacks = [
        # With the tiny dataset, the default 0.5 decision boundary is not
        # meaningful. Keep the epoch that best ranks held-out positives above
        # held-out noise, then estimate a provisional threshold below.
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    model.fit(X, y, validation_data=(vx, vy), epochs=args.epochs, batch_size=32, callbacks=callbacks, verbose=2)
    keras_path = args.out_dir / "ey_rocky.keras"
    tflite_path = args.out_dir / "ey_rocky.tflite"
    model.save(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path.write_bytes(converter.convert())

    scores = model.predict(vx, verbose=0).reshape(-1)
    thresholds = sorted(set([0.5, 0.7, 0.85, 0.95] + [float(x) for x in scores]))
    threshold_report = {}
    for threshold in thresholds:
        threshold_report[str(threshold)] = {
            "held_out_positive_recall": float(np.mean(scores[:len(val_pos)] >= threshold)) if val_pos else None,
            "noise_window_false_positive_rate": float(np.mean(scores[len(val_pos):] >= threshold)) if val_noise else None,
        }
    best_threshold = max(
        thresholds,
        key=lambda threshold: (
            (threshold_report[str(threshold)]["held_out_positive_recall"] or 0.0)
            + (1.0 - (threshold_report[str(threshold)]["noise_window_false_positive_rate"] or 0.0))
        ),
    )
    report = {
        "phrase": "ey rocky",
        "sample_rate_hz": SAMPLE_RATE,
        "input_shape": [1, CLIP_SAMPLES, 1],
        "input_type": "float32 PCM in [-1, 1]",
        "output": "wake probability [0, 1]",
        "unique_positive_clips": len(positives),
        "held_out_positive_clips": len(val_pos),
        "negative_source_seconds": len(noise_audio) / SAMPLE_RATE,
        "provisional_threshold_from_small_validation_set": best_threshold,
        "threshold_diagnostics": threshold_report,
        "limitations": ["prototype only", "threshold is tuned on only six held-out positive clips", "negative data contains noise only, not ordinary speech", "small single-session positive set"],
    }
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {tflite_path} ({tflite_path.stat().st_size:,} bytes)")
    print(f"Report: {args.out_dir / 'training_report.json'}")
    print("This model is experimental; do not treat noise-only validation as proof of real-world reliability.")


if __name__ == "__main__":
    main()
