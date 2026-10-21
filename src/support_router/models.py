import time
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from support_router.common import (
    PREPROCESSING,
    check_file,
    digest,
    file_info,
    read_json,
    write_json,
)
from support_router.data import normalize


def baseline(c, config):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", logistic(c, config)),
        ]
    )


def logistic(c, config):
    return LogisticRegression(
        C=c,
        max_iter=config["model"]["max_iter"],
        solver="lbfgs",
        random_state=config["project"]["seed"],
    )


def load_encoder(config, source):
    path = Path(config["project"]["data_dir"]) / source["encoder"]["path"]
    return SentenceTransformer(
        str(path), device="cpu", local_files_only=True, trust_remote_code=False
    )


def encoder_metadata(encoder, source):
    return {
        "id": source["encoder"]["id"],
        "revision": source["encoder"]["revision"],
        "max_seq_length": encoder.max_seq_length,
        "dimension": (
            encoder.get_embedding_dimension()
            if hasattr(encoder, "get_embedding_dimension")
            else encoder.get_sentence_embedding_dimension()
        ),
        "dtype": "float32",
        "normalize_embeddings": True,
        "preprocessing": PREPROCESSING,
        "tokenizer": encoder.tokenizer.__class__.__name__,
        "tokenizer_parameters": str(encoder.tokenizer.init_kwargs),
    }


def embedding_key(frame, metadata):
    return digest(
        {
            "encoder": metadata,
            "sample_ids": frame.sample_id.tolist(),
            "texts": frame.normalized.tolist(),
            "preprocessing": PREPROCESSING,
        }
    )


def encode(encoder, texts, batch_size=32):
    return encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def cached_embeddings(config, encoder, frame, metadata):
    key = embedding_key(frame, metadata)
    root = Path(config["project"]["data_dir"]) / "cache" / "embeddings"
    root.mkdir(parents=True, exist_ok=True)
    path, meta = root / f"{key}.npy", root / f"{key}.json"
    if path.exists() and meta.exists():
        info = read_json(meta)
        check_file(path, info["file"])
        array = np.load(path, allow_pickle=False)
    else:
        print(f"Encoding {len(frame)} records (cache {key[:10]})", flush=True)
        array = encode(encoder, frame.normalized, config["model"]["batch_size"])
        temp = path.with_suffix(".tmp.npy")
        np.save(temp, array, allow_pickle=False)
        temp.replace(path)
        write_json(meta, {"key": key, "encoder": metadata, "file": file_info(path)})
    if array.shape != (len(frame), metadata["dimension"]) or array.dtype != np.float32:
        raise ValueError("Invalid embedding cache shape/dtype")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite embeddings")
    return array


def predict_proba(model, family, texts, encoder=None):
    inputs = [normalize(x) for x in texts]
    return model.predict_proba(inputs if family == "tfidf" else encode(encoder, inputs))


def stable_top(probabilities, classes, k=3):
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(), 1, atol=1e-5):
        raise ValueError("Invalid classifier probabilities")
    # Stable sort preserves classifier.classes_ order on score ties.
    indices = np.argsort(-probabilities, kind="stable")[:k]
    return [{"category": str(classes[i]), "score": float(probabilities[i])} for i in indices]


def save_model(path, model):
    temporary = Path(path).with_suffix(".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)


def benchmark(call, texts):
    texts = list(texts)[:100]
    if len(texts) != 100:
        raise ValueError("Benchmark requires 100 fixed samples")
    for text in texts[:10]:
        call(text)
    durations = []
    for text in texts:
        start = time.perf_counter()
        call(text)
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "n": 100,
        "warmup": 10,
        "batch_size": 1,
        "http_included": False,
        "network": False,
        "input_characters": [len(x) for x in texts],
    }
