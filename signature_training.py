"""
Utilidades de entrenamiento para el modelo de reconocimiento de firmas.

Incluye:
- Carga y división de datos (64/16/20) sin modificar la normalización original.
- Generación eficiente de triplets con minería negativa dura en el batch.
- Bucle de entrenamiento optimizado con tf.data + tf.function.
- Cálculo de métricas (EER, separación) y guardado de artefactos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Configuración
# =============================================================================


@dataclass
class TrainingConfig:
    """Hyperparámetros y rutas del entrenamiento."""

    epochs: int = 150
    steps_per_epoch: int = 220
    val_steps: int = 40
    batch_size: int = 64
    hard_negatives: int = 4
    margin: float = 0.5
    learning_rate: float = 0.01
    lr_decay_steps: int = 400
    lr_decay_rate: float = 0.95
    dropout: float = 0.3
    l2_reg: float = 1e-4
    embedding_dim: int = 128
    seed: int | None = 42
    verbose: bool = True

    model_path: str = "embedding_network_final.h5"
    results_path: str = "lstm_stacked_final_results.json"
    cache_path: str = "lstm_stacked_final_SUCCESS.npz"

    x_path: str = "Task2_Preprocesado/X_features.npy"
    y_path: str = "Task2_Preprocesado/Y_user.npy"
    mask_path: str = "Task2_Preprocesado/M_mask.npy"


# =============================================================================
# Utilidades de datos
# =============================================================================


def _set_global_seed(seed: int | None) -> np.random.Generator:
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
        return np.random.default_rng(seed)
    return np.random.default_rng()


def load_dataset(config: TrainingConfig) -> Dict[str, np.ndarray]:
    """Carga y divide el dataset sin alterar la normalización existente."""
    X = np.load(config.x_path).astype(np.float32)
    Y = np.load(config.y_path).astype(np.int32)

    if Path(config.mask_path).exists():
        _ = np.load(config.mask_path)  # Solo para validar presencia; no se altera X.

    n_train = int(0.64 * len(X))
    n_val = int(0.16 * len(X))

    dataset = {
        "X_train": X[:n_train],
        "Y_train": Y[:n_train],
        "X_val": X[n_train : n_train + n_val],
        "Y_val": Y[n_train : n_train + n_val],
        "X_test": X[n_train + n_val :],
        "Y_test": Y[n_train + n_val :],
    }

    if config.verbose:
        print(f"  Dataset -> Train: {dataset['X_train'].shape}, "
              f"Val: {dataset['X_val'].shape}, Test: {dataset['X_test'].shape}")

    return dataset


# =============================================================================
# Generación de triplets
# =============================================================================


def _build_user_index(Y: np.ndarray) -> Dict[int, np.ndarray]:
    return {user: np.where(Y == user)[0] for user in np.unique(Y)}


def _triplet_batch_generator(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    hard_negatives: int,
    rng: np.random.Generator,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Genera batches infinitos de (anchor, positive, negative_pool)."""

    user_to_indices = _build_user_index(Y)
    users = np.array(list(user_to_indices.keys()))

    seq_len = X.shape[1]
    feat_dim = X.shape[2]

    while True:
        anchors = np.empty((batch_size, seq_len, feat_dim), dtype=np.float32)
        positives = np.empty_like(anchors)
        negatives = np.empty(
            (batch_size, hard_negatives, seq_len, feat_dim), dtype=np.float32
        )

        filled = 0
        while filled < batch_size:
            user = rng.choice(users)
            idxs = user_to_indices[user]
            if len(idxs) < 2:
                continue

            a_idx, p_idx = rng.choice(idxs, size=2, replace=False)
            other_users = users[users != user]

            if len(other_users) == 0:
                continue

            anchors[filled] = X[a_idx]
            positives[filled] = X[p_idx]

            for k in range(hard_negatives):
                neg_user = rng.choice(other_users)
                neg_idx = rng.choice(user_to_indices[neg_user])
                negatives[filled, k] = X[neg_idx]

            filled += 1

        yield anchors, positives, negatives


def make_triplet_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    hard_negatives: int,
    seed: int | None,
) -> tf.data.Dataset:
    """Crea un tf.data.Dataset infinito con batches de triplets."""

    rng = _set_global_seed(seed)

    generator = lambda: _triplet_batch_generator(
        X, Y, batch_size, hard_negatives, rng
    )

    output_signature = (
        tf.TensorSpec(shape=(batch_size, X.shape[1], X.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, X.shape[1], X.shape[2]), dtype=tf.float32),
        tf.TensorSpec(
            shape=(batch_size, hard_negatives, X.shape[1], X.shape[2]),
            dtype=tf.float32,
        ),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    options = tf.data.Options()
    options.autotune.enabled = True
    options.deterministic = False
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)


# =============================================================================
# Modelo y pérdida
# =============================================================================


def build_embedding_model(config: TrainingConfig) -> keras.Model:
    """Red LSTM apilada -> Dense -> embedding L2-normalizado."""
    inputs = keras.Input(shape=(208, 4), name="signature")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(
        128, return_sequences=True, dropout=config.dropout, recurrent_dropout=config.dropout
    )(x)
    x = layers.LSTM(
        64, return_sequences=False, dropout=config.dropout, recurrent_dropout=config.dropout
    )(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
    )(x)
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(
        config.embedding_dim,
        activation=None,
        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
    )(x)
    outputs = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), name="embedding")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="signature_encoder")

    if config.verbose:
        model.summary()

    return model


@tf.function
def triplet_loss(anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor, margin: float) -> tf.Tensor:
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    losses = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(losses)


# =============================================================================
# Métricas
# =============================================================================


def _pairwise_distances(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distances = squareform(pdist(embeddings, metric="euclidean"))
    genuine, impostor = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                genuine.append(distances[i, j])
            else:
                impostor.append(distances[i, j])
    return np.array(genuine, dtype=np.float32), np.array(impostor, dtype=np.float32)


def compute_eer(embeddings: np.ndarray, labels: np.ndarray, num_thresholds: int = 2000) -> Dict[str, float]:
    genuine, impostor = _pairwise_distances(embeddings, labels)
    max_distance = float(max(genuine.max(initial=0), impostor.max(initial=0), 1e-6))
    thresholds = np.linspace(0, max_distance, num_thresholds)

    best = {
        "eer": 1.0,
        "threshold": 0.0,
        "fnr": 1.0,
        "fpr": 1.0,
        "genuine_mean": float(genuine.mean()) if genuine.size else 0.0,
        "impostor_mean": float(impostor.mean()) if impostor.size else 0.0,
    }

    for threshold in thresholds:
        fnr = float((genuine > threshold).sum() / max(len(genuine), 1))
        fpr = float((impostor <= threshold).sum() / max(len(impostor), 1))
        eer = (fnr + fpr) / 2.0
        if eer < best["eer"]:
            best.update({"eer": eer, "threshold": threshold, "fnr": fnr, "fpr": fpr})

    best["separation"] = best["impostor_mean"] - best["genuine_mean"]
    return best


# =============================================================================
# Entrenamiento principal
# =============================================================================


def train_signature_model(config: TrainingConfig) -> Dict[str, float]:
    """Ejecuta entrenamiento + evaluación + guardado de artefactos."""

    print("=" * 80)
    print("ENTRENAMIENTO OPTIMIZADO - RECONOCIMIENTO DE FIRMAS")
    print("=" * 80)

    print("\n[1/5] Cargando datos...")
    data = load_dataset(config)

    print("\n[2/5] Construyendo modelo...")
    model = build_embedding_model(config)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    print("\n[3/5] Preparando generadores de triplets...")
    train_ds = make_triplet_dataset(
        data["X_train"], data["Y_train"], config.batch_size, config.hard_negatives, config.seed
    )
    val_ds = make_triplet_dataset(
        data["X_val"], data["Y_val"], config.batch_size, config.hard_negatives, (config.seed or 0) + 1
    )

    @tf.function
    def _select_hard_negatives(emb_anchor: tf.Tensor, emb_neg_pool: tf.Tensor) -> tf.Tensor:
        # emb_anchor: (B, D), emb_neg_pool: (B, K, D)
        dists = tf.norm(tf.expand_dims(emb_anchor, axis=1) - emb_neg_pool, axis=2)
        idx = tf.argmin(dists, axis=1, output_type=tf.int32)
        gather_idx = tf.stack([tf.range(tf.shape(idx)[0]), idx], axis=1)
        return tf.gather_nd(emb_neg_pool, gather_idx)

    @tf.function
    def train_step(anchor, positive, neg_pool):
        bsize = tf.shape(anchor)[0]
        hn = tf.shape(neg_pool)[1]
        neg_pool_flat = tf.reshape(neg_pool, (bsize * hn, anchor.shape[1], anchor.shape[2]))

        with tf.GradientTape() as tape:
            emb_a = model(anchor, training=True)
            emb_p = model(positive, training=True)
            emb_n_pool = model(neg_pool_flat, training=True)
            emb_n_pool = tf.reshape(emb_n_pool, (bsize, hn, config.embedding_dim))
            hard_neg = _select_hard_negatives(emb_a, emb_n_pool)
            loss = triplet_loss(emb_a, emb_p, hard_neg, config.margin)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    @tf.function
    def val_step(anchor, positive, neg_pool):
        bsize = tf.shape(anchor)[0]
        hn = tf.shape(neg_pool)[1]
        neg_pool_flat = tf.reshape(neg_pool, (bsize * hn, anchor.shape[1], anchor.shape[2]))

        emb_a = model(anchor, training=False)
        emb_p = model(positive, training=False)
        emb_n_pool = model(neg_pool_flat, training=False)
        emb_n_pool = tf.reshape(emb_n_pool, (bsize, hn, config.embedding_dim))
        hard_neg = _select_hard_negatives(emb_a, emb_n_pool)
        return triplet_loss(emb_a, emb_p, hard_neg, config.margin)

    print("\n[4/5] Entrenando...")
    print(f"  Epochs: {config.epochs} | Batch: {config.batch_size} | Hard negatives: {config.hard_negatives}")
    print("-" * 80)

    train_history, val_history = [], []
    for epoch in range(1, config.epochs + 1):
        epoch_losses = []
        for step, batch in enumerate(train_ds.take(config.steps_per_epoch), start=1):
            loss = train_step(*batch)
            epoch_losses.append(float(loss))

        val_losses = []
        for batch in val_ds.take(config.val_steps):
            v_loss = val_step(*batch)
            val_losses.append(float(v_loss))

        train_history.append(float(np.mean(epoch_losses)))
        val_history.append(float(np.mean(val_losses)))

        if epoch % max(1, config.epochs // 15) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{config.epochs} "
                f"- train_loss: {train_history[-1]:.5f} "
                f"- val_loss: {val_history[-1]:.5f}"
            )

    print("-" * 80)

    print("\n[5/5] Evaluando y guardando artefactos...")

    train_embeddings = model.predict(data["X_train"], batch_size=128, verbose=0)
    val_embeddings = model.predict(data["X_val"], batch_size=128, verbose=0)
    test_embeddings = model.predict(data["X_test"], batch_size=128, verbose=0)

    val_metrics = compute_eer(val_embeddings, data["Y_val"])
    test_metrics = compute_eer(test_embeddings, data["Y_test"])

    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)

    np.savez_compressed(
        config.cache_path,
        embeddings_train=train_embeddings,
        embeddings_val=val_embeddings,
        embeddings_test=test_embeddings,
        Y_train=data["Y_train"],
        Y_val=data["Y_val"],
        Y_test=data["Y_test"],
        train_losses=np.array(train_history, dtype=np.float32),
        val_losses=np.array(val_history, dtype=np.float32),
    )

    results = {
        "timestamp": tf.timestamp().numpy().item(),
        "config": asdict(config),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_final_loss": train_history[-1],
        "val_final_loss": val_history[-1],
        "model_params": int(model.count_params()),
        "embedding_dim": config.embedding_dim,
        "epochs": config.epochs,
        "margin": config.margin,
    }

    with open(config.results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Modelo: {config.model_path}")
    print(f"  ✓ Artefactos: {config.cache_path}")
    print(f"  ✓ Resultados: {config.results_path}")
    print(f"  ✓ Test EER: {test_metrics['eer'] * 100:.2f}% "
          f"(threshold={test_metrics['threshold']:.4f})")

    return results

