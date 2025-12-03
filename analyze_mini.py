"""
Análisis simplificado de resultados de train_mini.py con gráficos
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy import stats

# Config
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("="*80)
print("ANALISIS DE RESULTADOS: TRAIN_MINI.PY")
print("="*80)

# Cargar datos
print("\n[1/3] Cargando datos...")
with open("lstm_mini_results.json", "r") as f:
    results = json.load(f)

from signature_training import load_dataset, TrainingConfig
import tensorflow as tf

config = TrainingConfig(
    x_path="Task2_Preprocesado/X_features.npy",
    y_path="Task2_Preprocesado/Y_user.npy",
    mask_path="Task2_Preprocesado/M_mask.npy",
)

data = load_dataset(config)
X_train, Y_train = data["X_train"], data["Y_train"]
X_val, Y_val = data["X_val"], data["Y_val"]
X_test, Y_test = data["X_test"], data["Y_test"]

# Cargar modelo
try:
    model = tf.keras.models.load_model("embedding_network_mini.h5")
except:
    print("  [Reconstruyendo modelo...]")
    inputs = tf.keras.Input(shape=(208, 4), name="signature")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), name="embedding")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="signature_encoder")
    model.load_weights("embedding_network_mini.h5")

# Generar embeddings
print("[2/3] Generando embeddings...")
emb_train = model.predict(X_train, batch_size=64, verbose=0)
emb_val = model.predict(X_val, batch_size=64, verbose=0)
emb_test = model.predict(X_test, batch_size=64, verbose=0)

print("  Train embeddings shape: " + str(emb_train.shape))
print("  Val embeddings shape: " + str(emb_val.shape))
print("  Test embeddings shape: " + str(emb_test.shape))

# Calcular distancias
print("[3/3] Calculando distancias...")

def compute_distances(embeddings, labels):
    distances = squareform(pdist(embeddings, metric="euclidean"))
    genuine, impostor = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                genuine.append(distances[i, j])
            else:
                impostor.append(distances[i, j])
    return np.array(genuine), np.array(impostor)

gen_train, imp_train = compute_distances(emb_train, Y_train)
gen_val, imp_val = compute_distances(emb_val, Y_val)
gen_test, imp_test = compute_distances(emb_test, Y_test)

print("  Train: " + str(len(gen_train)) + " genuinas, " + str(len(imp_train)) + " impostoras")
print("  Val: " + str(len(gen_val)) + " genuinas, " + str(len(imp_val)) + " impostoras")
print("  Test: " + str(len(gen_test)) + " genuinas, " + str(len(imp_test)) + " impostoras")

# Crear tabla de metricas
metricas = []
datasets = ["Train", "Val", "Test"]
gen_lists = [gen_train, gen_val, gen_test]
imp_lists = [imp_train, imp_val, imp_test]

for ds, gen, imp in zip(datasets, gen_lists, imp_lists):
    sep = imp.mean() - gen.mean()
    overlap = len(np.where(gen > imp.mean())[0]) / len(gen) * 100
    metricas.append({
        "Dataset": ds,
        "Gen_mean": gen.mean(),
        "Gen_std": gen.std(),
        "Imp_mean": imp.mean(),
        "Imp_std": imp.std(),
        "Separacion": sep,
        "Overlap_pct": overlap
    })

df_metrics = pd.DataFrame(metricas)

print("\n" + "="*80)
print("METRICAS DE DISTANCIAS")
print("="*80)
print(df_metrics.to_string(index=False))

# ============================================================================
# CREAR GRAFICOS
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1. Histogramas (3x1)
for idx, (gen, imp, title, ds) in enumerate([
    (gen_train, imp_train, "TRAIN", "Train"),
    (gen_val, imp_val, "VALIDATION", "Val"),
    (gen_test, imp_test, "TEST", "Test"),
]):
    ax = plt.subplot(3, 3, idx + 1)
    ax.hist(gen, bins=40, alpha=0.6, label="Genuine (mean=%.4f)" % gen.mean(), color="green")
    ax.hist(imp, bins=40, alpha=0.6, label="Impostor (mean=%.4f)" % imp.mean(), color="red")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Frequency")
    ax.set_title(title + ": Distance Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

# 2. Box plots (3x1)
ax = plt.subplot(3, 3, 4)
data_box = [gen_train, imp_train, gen_val, imp_val, gen_test, imp_test]
labels_box = ["Train\nGenuine", "Train\nImpostor", "Val\nGenuine", "Val\nImpostor", 
              "Test\nGenuine", "Test\nImpostor"]
bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True)
for idx, (patch, label) in enumerate(zip(bp["boxes"], labels_box)):
    if "Genuine" in label:
        patch.set_facecolor("lightgreen")
    else:
        patch.set_facecolor("lightcoral")
ax.set_ylabel("Euclidean Distance")
ax.set_title("BOX PLOT: All Distances")
ax.grid(True, alpha=0.3, axis="y")

# 3. Violin plot
ax = plt.subplot(3, 3, 5)
parts = ax.violinplot([gen_test, imp_test], positions=[1, 2], showmeans=True, showmedians=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Genuine", "Impostor"])
ax.set_ylabel("Distance")
ax.set_title("TEST: Violin Plot")
ax.grid(True, alpha=0.3, axis="y")

# 4. Separation metric
ax = plt.subplot(3, 3, 6)
separation_vals = df_metrics["Separacion"].values
colors_sep = ["green" if x > 0.01 else "orange" if x > 0.005 else "red" for x in separation_vals]
ax.bar(df_metrics["Dataset"], separation_vals, color=colors_sep, alpha=0.7)
ax.axhline(y=0.01, color="green", linestyle="--", linewidth=1, label="Good")
ax.axhline(y=0.005, color="orange", linestyle="--", linewidth=1, label="Fair")
ax.set_ylabel("Separation (Impostor_mean - Genuine_mean)")
ax.set_title("SEPARATION QUALITY")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 5. Overlap percentage
ax = plt.subplot(3, 3, 7)
overlap_vals = df_metrics["Overlap_pct"].values
colors_overlap = ["green" if x < 20 else "orange" if x < 40 else "red" for x in overlap_vals]
ax.bar(df_metrics["Dataset"], overlap_vals, color=colors_overlap, alpha=0.7)
ax.set_ylabel("Overlap Percentage (%)")
ax.set_title("OVERLAP: Genuine above Impostor_mean")
ax.grid(True, alpha=0.3, axis="y")

# 6. CDF curves
ax = plt.subplot(3, 3, 8)
for gen, imp, ds in zip([gen_train, gen_val, gen_test], [imp_train, imp_val, imp_test], datasets):
    sorted_gen = np.sort(gen)
    sorted_imp = np.sort(imp)
    ax.plot(sorted_gen, np.arange(1, len(sorted_gen)+1)/len(sorted_gen), 
            label=ds+" Genuine", linewidth=2)
    ax.plot(sorted_imp, np.arange(1, len(sorted_imp)+1)/len(sorted_imp), 
            label=ds+" Impostor", linewidth=2, linestyle="--")
ax.set_xlabel("Euclidean Distance")
ax.set_ylabel("Cumulative Probability")
ax.set_title("CDF: Cumulative Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 7. Std comparison
ax = plt.subplot(3, 3, 9)
x_pos = np.arange(len(df_metrics))
width = 0.2
ax.bar(x_pos - width, df_metrics["Gen_std"], width, label="Genuine std", alpha=0.8)
ax.bar(x_pos, df_metrics["Imp_std"], width, label="Impostor std", alpha=0.8)
ax.set_ylabel("Standard Deviation")
ax.set_title("VARIABILITY")
ax.set_xticks(x_pos)
ax.set_xticklabels(df_metrics["Dataset"])
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("analisis_distancias.png", dpi=300, bbox_inches="tight")
print("\n[GRAFICOS GUARDADOS]")
print("  Archivo: analisis_distancias.png")

# ============================================================================
# SEGUNDO CONJUNTO DE GRAFICOS - METRICAS Y CURVAS
# ============================================================================

fig2 = plt.figure(figsize=(16, 10))

# 1. Normal distribution fit
ax = plt.subplot(2, 3, 1)
ax.hist(gen_test, bins=30, density=True, alpha=0.5, label="Genuine (Real)", color="green")
ax.hist(imp_test, bins=30, density=True, alpha=0.5, label="Impostor (Real)", color="red")

mu_gen, sigma_gen = gen_test.mean(), gen_test.std()
mu_imp, sigma_imp = imp_test.mean(), imp_test.std()
x = np.linspace(0, max(gen_test.max(), imp_test.max()), 100)
ax.plot(x, stats.norm.pdf(x, mu_gen, sigma_gen), 'g-', linewidth=2, label="Normal Genuine")
ax.plot(x, stats.norm.pdf(x, mu_imp, sigma_imp), 'r-', linewidth=2, label="Normal Impostor")
ax.set_xlabel("Euclidean Distance")
ax.set_ylabel("Probability Density")
ax.set_title("TEST: Normal Distribution Fit")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. ROC-like curve: FPR vs FNR
ax = plt.subplot(2, 3, 2)

# Encontrar threshold optimo
thresholds = np.linspace(0, max(gen_test.max(), imp_test.max()), 200)
fpr_list, fnr_list, eer_list = [], [], []
best_eer = 1.0
best_threshold = 0.0

for threshold in thresholds:
    fpr = np.sum(imp_test <= threshold) / len(imp_test)
    fnr = np.sum(gen_test > threshold) / len(gen_test)
    eer = (fnr + fpr) / 2.0
    fpr_list.append(fpr)
    fnr_list.append(fnr)
    eer_list.append(eer)
    if eer < best_eer:
        best_eer = eer
        best_threshold = threshold

ax.plot(thresholds, fpr_list, label="FPR (False Positive Rate)", linewidth=2)
ax.plot(thresholds, fnr_list, label="FNR (False Negative Rate)", linewidth=2)
ax.axvline(x=best_threshold, color="green", linestyle="--", linewidth=2, label="Optimal Threshold")
ax.set_xlabel("Distance Threshold")
ax.set_ylabel("Error Rate")
ax.set_title("TEST: FPR vs FNR (EER=%.2f%%)" % (best_eer*100))
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Metrics table
ax = plt.subplot(2, 3, 3)
ax.axis("tight")
ax.axis("off")

table_data = []
for idx, row in df_metrics.iterrows():
    table_data.append([
        row["Dataset"],
        "%.4f" % row["Gen_mean"],
        "%.4f" % row["Imp_mean"],
        "%.4f" % row["Separacion"],
        "%.1f%%" % row["Overlap_pct"]
    ])

table = ax.table(cellText=table_data,
                colLabels=["Dataset", "Gen Mean", "Imp Mean", "Separation", "Overlap"],
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(5):
    table.get_children()[i].set_facecolor("#40466e")
    table.get_children()[i].set_text_props(weight="bold", color="white")

ax.set_title("METRICS SUMMARY", fontsize=12, weight="bold", pad=20)

# 4. Confusion matrix for test
ax = plt.subplot(2, 3, 4)

tp = np.sum(imp_test <= best_threshold)
fp = np.sum(gen_test <= best_threshold)
tn = np.sum(gen_test > best_threshold)
fn = np.sum(imp_test > best_threshold)

cm_data = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm_data, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Genuine", "Pred Impostor"])
ax.set_yticklabels(["True Genuine", "True Impostor"])
ax.set_title("CONFUSION MATRIX (TEST)\nThreshold=%.4f" % best_threshold)

for i in range(2):
    for j in range(2):
        val = cm_data[i, j]
        color = "white" if val > cm_data.max()/2 else "black"
        text = ax.text(j, i, int(val), ha="center", va="center", 
                      color=color, fontsize=14, weight="bold")

# 5. Performance metrics
ax = plt.subplot(2, 3, 5)

accuracy = (tn + tp) / (tn + tp + fn + fp)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

metrics_values = [accuracy, precision, recall, f1, best_eer]
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "EER"]
colors_perf = ["green" if x > 0.8 else "orange" if x > 0.6 else "red" for x in metrics_values]

bars = ax.bar(metrics_names, metrics_values, color=colors_perf, alpha=0.7)
ax.set_ylabel("Score")
ax.set_title("TEST PERFORMANCE METRICS")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           '%.2f' % val, ha='center', va='bottom', fontweight='bold')

# 6. Summary statistics
ax = plt.subplot(2, 3, 6)
ax.axis("tight")
ax.axis("off")

summary_text = (
    "TRAINING SUMMARY\n"
    "============================\n\n"
    "Configuration:\n"
    "  Epochs: 5\n"
    "  Batch Size: 16\n"
    "  Margin: 0.5\n"
    "  Learning Rate: 0.01\n"
    "  Hard Negatives: 2\n"
    "  Data Used: 10 percent of 1600 samples\n\n"
    "TEST RESULTS:\n"
    "  EER Optimal: %.2f percent\n"
    "  Threshold: %.4f\n"
    "  Accuracy: %.2f percent\n"
    "  Precision: %.2f\n"
    "  Recall: %.2f\n"
    "  F1-Score: %.4f\n\n"
    "SEPARATION:\n"
    "  Genuine Distance: %.4f plus-minus %.4f\n"
    "  Impostor Distance: %.4f plus-minus %.4f\n"
    "  Separation: %.4f\n"
    "  Overlap: %.1f percent\n\n"
    "CLASSIFICATION:\n"
    "  True Positives: %d\n"
    "  False Positives: %d\n"
    "  True Negatives: %d\n"
    "  False Negatives: %d\n"
) % (best_eer*100, best_threshold, accuracy*100, precision, recall, f1,
     gen_test.mean(), gen_test.std(), imp_test.mean(), imp_test.std(),
     imp_test.mean() - gen_test.mean(), df_metrics.iloc[2]["Overlap_pct"],
     tp, fp, tn, fn)

ax.text(0.05, 0.5, summary_text, transform=ax.transAxes, fontsize=9,
       verticalalignment="center", fontfamily="monospace",
       bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("analisis_metricas_rendimiento.png", dpi=300, bbox_inches="tight")
print("  Archivo: analisis_metricas_rendimiento.png")

# ============================================================================
# GUARDAR DATAFRAMES EN CSV
# ============================================================================

df_metrics.to_csv("metricas_resumen.csv", index=False)

# DataFrame detallado
distances_data = {
    "Dataset": (["Train"]*len(gen_train) + ["Train"]*len(imp_train) + 
               ["Val"]*len(gen_val) + ["Val"]*len(imp_val) + 
               ["Test"]*len(gen_test) + ["Test"]*len(imp_test)),
    "Type": (["Genuine"]*len(gen_train) + ["Impostor"]*len(imp_train) +
            ["Genuine"]*len(gen_val) + ["Impostor"]*len(imp_val) +
            ["Genuine"]*len(gen_test) + ["Impostor"]*len(imp_test)),
    "Distance": list(gen_train) + list(imp_train) + 
               list(gen_val) + list(imp_val) + 
               list(gen_test) + list(imp_test)
}

df_distances = pd.DataFrame(distances_data)
df_distances.to_csv("distancias_detalladas.csv", index=False)

print("  Archivo: metricas_resumen.csv")
print("  Archivo: distancias_detalladas.csv")

# ============================================================================
# MOSTRAR RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("EVALUACION FINAL (TEST SET)")
print("="*80)
print("EER Optimo: %.2f%%" % (best_eer*100))
print("Threshold Optimo: %.4f" % best_threshold)
print("Accuracy: %.2f%%" % (accuracy*100))
print("Precision: %.4f" % precision)
print("Recall: %.4f" % recall)
print("F1-Score: %.4f" % f1)

print("\nSeparacion:")
print("  Genuinas - Media: %.4f, Std: %.4f" % (gen_test.mean(), gen_test.std()))
print("  Impostoras - Media: %.4f, Std: %.4f" % (imp_test.mean(), imp_test.std()))
print("  Diferencia: %.4f" % (imp_test.mean() - gen_test.mean()))
print("  Overlap: %.1f%%" % df_metrics.iloc[2]["Overlap_pct"])

if df_metrics.iloc[2]["Separacion"] > 0.01:
    eval_text = "[BUENA] Separacion clara entre genuinas e impostoras"
elif df_metrics.iloc[2]["Separacion"] > 0.005:
    eval_text = "[ACEPTABLE] Separacion presente pero mejorable"
else:
    eval_text = "[POBRE] Muy poca separacion, requiere optimizacion"

print("\nEvaluacion: " + eval_text)

print("\n" + "="*80)
print("Analisis completado. Revisa los archivos PNG para visualizaciones.")
print("="*80)
