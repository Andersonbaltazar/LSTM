"""
Entrenamiento rápido para depuración usando la canalización optimizada.
"""

from signature_training import TrainingConfig, train_signature_model


if __name__ == "__main__":
    config = TrainingConfig(
        epochs=60,
        steps_per_epoch=120,
        val_steps=30,
        batch_size=48,
        hard_negatives=2,
        margin=0.4,
        learning_rate=0.006,
        lr_decay_steps=280,
        lr_decay_rate=0.9,
        model_path="embedding_network_final.h5",
        results_path="lstm_stacked_final_results.json",
        cache_path="lstm_stacked_final_SUCCESS.npz",
        seed=3,
    )

    train_signature_model(config)
