#!/usr/bin/env python3
"""Entrenamiento baseline (margin=0.1) para el modelo de firmas."""

from signature_training import TrainingConfig, train_signature_model


if __name__ == "__main__":
    config = TrainingConfig(
        epochs=100,
        steps_per_epoch=180,
        val_steps=45,
        batch_size=48,
        hard_negatives=2,
        margin=0.1,
        learning_rate=0.001,
        lr_decay_steps=256,
        lr_decay_rate=0.95,
        model_path="embedding_network_final.h5",
        results_path="lstm_stacked_final_results.json",
        cache_path="lstm_stacked_final_SUCCESS.npz",
        seed=7,
    )

    train_signature_model(config)
