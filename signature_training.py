%%writefile /content/sample_data/signature_training.py
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass
from typing import Dict, Generator

@dataclass
class TrainingConfig:
    """Configuración para el entrenamiento"""
    epochs: int = 180
    steps_per_epoch: int = 240
    val_steps: int = 60
    batch_size: int = 64
    hard_negatives: int = 4
    margin: float = 0.25
    learning_rate: float = 0.01
    lr_decay_steps: int = 450
    lr_decay_rate: float = 0.94
    embedding_dim: int = 256
    model_path: str = "embedding_network_final.h5"
    results_path: str = "lstm_stacked_final_results.json"
    cache_path: str = "lstm_stacked_final_SUCCESS.npz"
    x_path: str = "Task2_Preprocesado/X_features.npy"
    y_path: str = "Task2_Preprocesado/Y_user.npy"
    mask_path: str = "Task2_Preprocesado/M_mask.npy"
    seed: int = 42

def load_dataset(config: TrainingConfig) -> Dict:
    np.random.seed(config.seed)
    X = np.load(config.x_path).astype(np.float32)
    Y = np.load(config.y_path).astype(np.int32)
    X_padded = np.zeros((len(X), 400, 4), dtype=np.float32)
    X_padded[:, :X.shape[1], :] = X
    X = X_padded
    n = len(X)
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.7*n)]
    val_idx = idx[int(0.7*n):int(0.85*n)]
    return {"X_train": X[train_idx], "Y_train": Y[train_idx], "X_val": X[val_idx], "Y_val": Y[val_idx]}

def build_encoder(embedding_dim: int, input_shape=(400, 4)) -> keras.Model:
    signature_input = keras.Input(shape=input_shape, name="signature_input")
    x = layers.Masking(mask_value=0.0)(signature_input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim)(x)
    embedding_output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2_normalize")(x)
    return keras.Model(inputs=signature_input, outputs=embedding_output, name="signature_encoder")

class SiameseModel(keras.Model):
    def __init__(self, encoder, margin=0.25):
        super().__init__()
        self.encoder = encoder
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.encoder(inputs)

    def train_step(self, data):
        inputs, _ = data
        with tf.GradientTape() as tape:
            emb_a = self.encoder(inputs["anchor_input"])
            emb_p = self.encoder(inputs["positive_input"])
            emb_n = self.encoder(inputs["negative_input"])
            pos_dist = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
            neg_dist = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
            loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + self.margin, 0.0))
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        inputs, _ = data
        emb_a = self.encoder(inputs["anchor_input"])
        emb_p = self.encoder(inputs["positive_input"])
        emb_n = self.encoder(inputs["negative_input"])
        pos_dist = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
        neg_dist = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
        loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + self.margin, 0.0))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

class TripletDataGenerator:
    def __init__(self, X: np.ndarray, Y: np.ndarray, batch_size: int):
        self.X, self.Y, self.batch_size = X, Y, batch_size
        self.unique_users = np.unique(Y)
        self.indices_by_user = {user: np.where(Y == user)[0] for user in self.unique_users}

    def __call__(self) -> Generator:
        while True:
            batch_anchors, batch_positives, batch_negatives = [], [], []
            for _ in range(self.batch_size):
                anchor_user = np.random.choice(self.unique_users)
                user_indices = self.indices_by_user[anchor_user]
                if len(user_indices) < 2: continue
                a_idx, p_idx = np.random.choice(user_indices, 2, replace=False)
                neg_user = np.random.choice([u for u in self.unique_users if u != anchor_user])
                n_idx = np.random.choice(self.indices_by_user[neg_user])
                batch_anchors.append(self.X[a_idx]); batch_positives.append(self.X[p_idx]); batch_negatives.append(self.X[n_idx])
            yield {"anchor_input": np.array(batch_anchors), "positive_input": np.array(batch_positives), "negative_input": np.array(batch_negatives)}, np.zeros(self.batch_size)

def train_signature_model(config: TrainingConfig):
    data = load_dataset(config)
    encoder = build_encoder(config.embedding_dim)
    siamese_model = SiameseModel(encoder, margin=config.margin)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.learning_rate, config.lr_decay_steps, config.lr_decay_rate)
    siamese_model.compile(optimizer=keras.optimizers.Adam(lr_schedule))
    train_gen = TripletDataGenerator(data["X_train"], data["Y_train"], config.batch_size)
    val_gen = TripletDataGenerator(data["X_val"], data["Y_val"], config.batch_size)
    siamese_model.fit(train_gen(), steps_per_epoch=config.steps_per_epoch, validation_data=val_gen(), validation_steps=config.val_steps, epochs=config.epochs)
    encoder.save(config.model_path)