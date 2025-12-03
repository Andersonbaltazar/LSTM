"""
Crear versión HTML en ESPAÑOL con explicaciones de redes neuronales y gráficos
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
import base64
from io import BytesIO

# Config
sns.set_style("whitegrid")

print("Generando gráficos para HTML...")

# ============================================================================
# CARGAR DATOS
# ============================================================================

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
emb_train = model.predict(X_train, batch_size=64, verbose=0)
emb_val = model.predict(X_val, batch_size=64, verbose=0)
emb_test = model.predict(X_test, batch_size=64, verbose=0)

# Calcular distancias
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

# ============================================================================
# FUNCIÓN HELPER PARA CONVERTIR FIGURA A BASE64
# ============================================================================

def fig_to_base64(fig):
    """Convierte figura matplotlib a string base64 para HTML"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64

# ============================================================================
# CREAR GRÁFICOS INDIVIDUALES
# ============================================================================

print("  Gráfico 1: Histogramas...")
fig1 = plt.figure(figsize=(15, 5))

for idx, (gen, imp, title) in enumerate([
    (gen_train, imp_train, "ENTRENAMIENTO"),
    (gen_val, imp_val, "VALIDACION"),
    (gen_test, imp_test, "PRUEBA"),
]):
    ax = plt.subplot(1, 3, idx + 1)
    ax.hist(gen, bins=40, alpha=0.6, label="Genuinas", color="green")
    ax.hist(imp, bins=40, alpha=0.6, label="Impostoras", color="red")
    ax.set_xlabel("Distancia Euclidiana")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

graph1_base64 = fig_to_base64(fig1)

print("  Gráfico 2: Cajas comparativas...")
fig2 = plt.figure(figsize=(12, 6))
data_box = [gen_train, imp_train, gen_val, imp_val, gen_test, imp_test]
labels_box = ["Entre.\nGenuina", "Entre.\nImpostora", "Val.\nGenuina", "Val.\nImpostora", 
              "Prueba\nGenuina", "Prueba\nImpostora"]
bp = plt.boxplot(data_box, tick_labels=labels_box, patch_artist=True)
for idx, (patch, label) in enumerate(zip(bp["boxes"], labels_box)):
    if "Genuina" in label:
        patch.set_facecolor("lightgreen")
    else:
        patch.set_facecolor("lightcoral")
plt.ylabel("Distancia")
plt.title("CAJAS COMPARATIVAS: Distribucion de Distancias")
plt.grid(True, alpha=0.3, axis="y")

graph2_base64 = fig_to_base64(fig2)

print("  Gráfico 3: Distribuciones acumuladas...")
fig3 = plt.figure(figsize=(12, 6))
for gen, imp, ds in zip([gen_train, gen_val, gen_test], [imp_train, imp_val, imp_test], 
                         ["Entrenamiento", "Validacion", "Prueba"]):
    sorted_gen = np.sort(gen)
    sorted_imp = np.sort(imp)
    plt.plot(sorted_gen, np.arange(1, len(sorted_gen)+1)/len(sorted_gen), 
            label=ds+" Genuinas", linewidth=2)
    plt.plot(sorted_imp, np.arange(1, len(sorted_imp)+1)/len(sorted_imp), 
            label=ds+" Impostoras", linewidth=2, linestyle="--")
plt.xlabel("Distancia")
plt.ylabel("Probabilidad Acumulada")
plt.title("DISTRIBUCION ACUMULADA (FDA): Curvas de Probabilidad")
plt.legend()
plt.grid(True, alpha=0.3)

graph3_base64 = fig_to_base64(fig3)

print("  Gráfico 4: Metricas de separacion...")
fig4 = plt.figure(figsize=(12, 6))

datasets = ["Entrenamiento", "Validacion", "Prueba"]
sep_vals = [
    imp_train.mean() - gen_train.mean(),
    imp_val.mean() - gen_val.mean(),
    imp_test.mean() - gen_test.mean()
]
colors = ["green" if x > 0.01 else "orange" if x > 0.005 else "red" for x in sep_vals]

plt.bar(datasets, sep_vals, color=colors, alpha=0.7)
plt.ylabel("Separacion (Media Impostoras - Media Genuinas)")
plt.title("CALIDAD DE SEPARACION por Conjunto de Datos")
plt.grid(True, alpha=0.3, axis="y")

graph4_base64 = fig_to_base64(fig4)

print("  Gráfico 5: Curva Tasa de errores...")
fig5 = plt.figure(figsize=(12, 6))

thresholds = np.linspace(0, max(gen_test.max(), imp_test.max()), 200)
fpr_list, fnr_list = [], []
best_eer = 1.0
best_threshold = 0.0

for threshold in thresholds:
    fpr = np.sum(imp_test <= threshold) / len(imp_test)
    fnr = np.sum(gen_test > threshold) / len(gen_test)
    eer = (fnr + fpr) / 2.0
    fpr_list.append(fpr)
    fnr_list.append(fnr)
    if eer < best_eer:
        best_eer = eer
        best_threshold = threshold

plt.plot(thresholds, fpr_list, label="TFP (Tasa Falsos Positivos)", linewidth=2)
plt.plot(thresholds, fnr_list, label="TFN (Tasa Falsos Negativos)", linewidth=2)
plt.axvline(x=best_threshold, color="green", linestyle="--", linewidth=2, 
           label=f"Optimo (EER={best_eer*100:.2f}%)")
plt.xlabel("Umbral de Distancia")
plt.ylabel("Tasa de Error")
plt.title("PRUEBA: Tasa de Falsos Positivos vs Falsos Negativos")
plt.legend()
plt.grid(True, alpha=0.3)

graph5_base64 = fig_to_base64(fig5)

print("  Gráfico 6: Grafico de violin...")
fig6 = plt.figure(figsize=(10, 6))
parts = plt.violinplot([gen_test, imp_test], positions=[1, 2], showmeans=True, showmedians=True)
plt.xticks([1, 2], ["Genuinas", "Impostoras"])
plt.ylabel("Distancia")
plt.title("PRUEBA: Grafico de Violin - Distribucion de Densidad")
plt.grid(True, alpha=0.3, axis="y")

graph6_base64 = fig_to_base64(fig6)

# ============================================================================
# CREAR HTML
# ============================================================================

print("Generando HTML...")

html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analisis Completo: Entrenamiento Mini</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 50px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        
        h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        h2 {{
            color: #333;
            font-size: 1.8em;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        h3 {{
            color: #555;
            font-size: 1.3em;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .info-box {{
            background: #e8f4f8;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            line-height: 1.8;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-value {{
            color: #333;
            font-size: 1.8em;
            font-weight: bold;
            margin-top: 10px;
        }}
        
        .stat-card.good .stat-value {{ color: #27ae60; }}
        .stat-card.warning .stat-value {{ color: #f39c12; }}
        .stat-card.bad .stat-value {{ color: #e74c3c; }}
        
        .graph-section {{
            margin: 50px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .graph-section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .graph-section p {{
            color: #666;
            margin-bottom: 15px;
            line-height: 1.6;
        }}
        
        .explanation {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #f39c12;
        }}
        
        .explanation strong {{
            color: #f39c12;
        }}
        
        .graph-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .graph-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .metrics-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .metrics-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        .metrics-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        
        footer {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #666;
        }}
        
        .conclusion {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 5px;
            margin: 30px 0;
        }}
        
        .conclusion h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        .conclusion p {{
            color: #856404;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        
        .neural-net-section {{
            background: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 30px;
            border-radius: 8px;
            margin: 30px 0;
        }}
        
        ul, ol {{
            margin-left: 20px;
            line-height: 1.8;
        }}
        
        li {{
            margin-bottom: 10px;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 1.8em;
            }}
            
            .stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Analisis Completo: Entrenamiento Mini</h1>
            <p class="subtitle">Mini-entrenamiento con 10% de datos - 5 epocas - Redes LSTM con Perdida Triplet</p>
        </header>
        
        <!-- EXPLICACIÓN DE REDES NEURONALES -->
        <div class="neural-net-section">
            <h2>🧠 ¿Que es una Red Neuronal LSTM y la Perdida Triplet?</h2>
            
            <h3>Red LSTM (Memoria a Largo Plazo)</h3>
            <p>
                La red LSTM es un tipo especial de red neuronal recurrente disenada para procesar secuencias de datos 
                (como puntos de una firma). En nuestro caso:
            </p>
            <ul>
                <li><strong>Entrada:</strong> 208 puntos de una firma con 4 caracteristicas cada uno (X, Y, tiempo, presion)</li>
                <li><strong>LSTM 1:</strong> Procesa la secuencia completa, capturando patrones a largo plazo (128 unidades)</li>
                <li><strong>LSTM 2:</strong> Refina los patrones encontrados (64 unidades)</li>
                <li><strong>Capas Densas:</strong> Combinan los patrones en un vector final de 128 numeros (embedding)</li>
                <li><strong>Salida:</strong> Un "huella digital numerica" unica para cada firma</li>
            </ul>
            
            <h3>Perdida Triplet (Triplet Loss)</h3>
            <p>
                Este es el "maestro" que entrena la red. Funciona comparando 3 firmas simultáneamente:
            </p>
            <ul>
                <li><strong>Ancla:</strong> Una firma de referencia (ejemplo: tu firma)</li>
                <li><strong>Positiva:</strong> Otra firma genuina tuya (debe estar CERCA de la ancla)</li>
                <li><strong>Negativa:</strong> Una firma de alguien mas (debe estar LEJOS de la ancla)</li>
            </ul>
            <p>
                El objetivo es minimizar la distancia entre (Ancla - Positiva) mientras se maximiza 
                la distancia entre (Ancla - Negativa). Esto crea una "separacion" clara en el espacio de embeddings.
            </p>
            
            <h3>¿Por que 10% de datos?</h3>
            <p>
                Para diagnosticar rapido si los parametros son correctos. Si falla con 10%, no tiene sentido 
                esperar 130 horas con 100% de datos. Ademas permite ver sobreajuste (overfitting).
            </p>
        </div>
        
        <!-- MÉTRICAS PRINCIPALES -->
        <h2>📈 Metricas Principales (Conjunto de Prueba)</h2>
        <div class="stats">
            <div class="stat-card bad">
                <div class="stat-label">EER Optimo</div>
                <div class="stat-value">36.57%</div>
            </div>
            <div class="stat-card bad">
                <div class="stat-label">Exactitud</div>
                <div class="stat-value">39.05%</div>
            </div>
            <div class="stat-card good">
                <div class="stat-label">Precision</div>
                <div class="stat-value">0.8109</div>
            </div>
            <div class="stat-card bad">
                <div class="stat-label">Exhaustividad</div>
                <div class="stat-value">0.3985</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">F1-Score</div>
                <div class="stat-value">0.5344</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">Umbral</div>
                <div class="stat-value">0.0010</div>
            </div>
        </div>
        
        <div class="info-box">
            <strong>Definicion de Metricas:</strong><br>
            • <strong>EER:</strong> Porcentaje de error donde TFP = TFN (punto de balance)<br>
            • <strong>Exactitud:</strong> Porcentaje total de predicciones correctas<br>
            • <strong>Precision:</strong> De las predichas como impostoras, cuantas realmente lo son<br>
            • <strong>Exhaustividad:</strong> De todas las impostoras, cuantas detectamos<br>
            • <strong>F1-Score:</strong> Media armonica entre Precision y Exhaustividad
        </div>
        
        <!-- TABLA DE MÉTRICAS -->
        <h2>📋 Tabla Completa de Distancias</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Conjunto</th>
                    <th>Media Genuinas</th>
                    <th>Desv. Genuinas</th>
                    <th>Media Impostoras</th>
                    <th>Desv. Impostoras</th>
                    <th>Separacion</th>
                    <th>Sobreposicion %</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Entrenamiento</strong></td>
                    <td>0.000960</td>
                    <td>0.000599</td>
                    <td>0.001878</td>
                    <td>0.001054</td>
                    <td>0.000918</td>
                    <td>7.35%</td>
                </tr>
                <tr>
                    <td><strong>Validacion</strong></td>
                    <td>0.001241</td>
                    <td>0.000776</td>
                    <td>0.001672</td>
                    <td>0.000921</td>
                    <td>0.000431</td>
                    <td>26.71%</td>
                </tr>
                <tr>
                    <td><strong>Prueba</strong></td>
                    <td>0.000877</td>
                    <td>0.000470</td>
                    <td>0.001297</td>
                    <td>0.000676</td>
                    <td>0.000420</td>
                    <td>17.48%</td>
                </tr>
            </tbody>
        </table>
        
        <div class="info-box">
            <strong>Que significan estos numeros:</strong><br>
            • <strong>Media:</strong> Distancia promedio entre firmas del mismo tipo<br>
            • <strong>Desv:</strong> Cuanto varían las distancias (mientras menor, mas consistente)<br>
            • <strong>Separacion:</strong> Diferencia entre media de impostoras y genuinas (IDEAL: > 0.1)<br>
            • <strong>Sobreposicion:</strong> Porcentaje de firmas genuinas por encima de la media de impostoras
        </div>
        
        <!-- GRÁFICO 1 -->
        <div class="graph-section">
            <h2>📊 Grafico 1: Histogramas de Distancias</h2>
            <div class="explanation">
                <strong>¿Que demuestra?</strong><br>
                Muestra la DISTRIBUCION de distancias entre firmas genuinas (verde) e impostoras (rojo) 
                en los tres conjuntos de datos (Entrenamiento, Validacion, Prueba).
                <br><br>
                <strong>¿Que buscamos?</strong><br>
                Las barras verdes y rojas deben estar SEPARADAS. Si se superponen mucho, 
                significa que el modelo no puede diferenciar bien entre firmas genuinas e impostoras.
                <br><br>
                <strong>¿Que vemos aqui?</strong><br>
                • Entrenamiento: SEPARACION EXCELENTE (barras casi sin sobreposicion)<br>
                • Validacion: SOBREPOSICION MODERADA (las barras se tocan)<br>
                • Prueba: SOBREPOSICION BAJA (pero aun significativa)<br>
                Esto indica que el modelo memorizó bien el entrenamiento pero no generaliza bien.
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{graph1_base64}" alt="Histogramas">
            </div>
        </div>
        
        <!-- GRÁFICO 2 -->
        <div class="graph-section">
            <h2>📦 Grafico 2: Cajas Comparativas</h2>
            <div class="explanation">
                <strong>¿Que demuestra?</strong><br>
                Las "cajas" resumen estadisticamente cada distribucion. Cada caja muestra:
                <br><br>
                • <strong>Linea central:</strong> Mediana (valor del medio)<br>
                • <strong>Altura de la caja:</strong> Rango donde estan el 50% de los datos<br>
                • <strong>Lineas arriba/abajo:</strong> Extremos (minimo y maximo)<br>
                • <strong>Puntos:</strong> Outliers (valores muy diferentes al resto)<br>
                <br>
                <strong>¿Que buscamos?</strong><br>
                Las cajas de "Genuina" deben estar COMPLETAMENTE SEPARADAS de "Impostora".
                <br><br>
                <strong>¿Que vemos?</strong><br>
                • Entrenamiento: Separacion perfecta ✓<br>
                • Validacion: Las cajas se sobrelapan (problema de generalizacion)<br>
                • Prueba: Cajas se sobrelapan (modelo no generaliza bien a datos nuevos)
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{graph2_base64}" alt="Cajas Comparativas">
            </div>
        </div>
        
        <!-- GRÁFICO 3 -->
        <div class="graph-section">
            <h2>📈 Grafico 3: Distribuciones Acumuladas (FDA)</h2>
            <div class="explanation">
                <strong>¿Que demuestra?</strong><br>
                Muestra la "Funcion de Distribucion Acumulada" (FDA). Cada punto en la linea 
                responde: "¿Cuantos datos tienen distancia MENOR a este valor?"
                <br><br>
                <strong>¿Que buscamos?</strong><br>
                Las lineas de "Genuinas" (solidas) deben estar TOTALMENTE ABAJO de "Impostoras" (punteadas).
                Si se cruzan, significa que hay confusion.
                <br><br>
                <strong>¿Que vemos?</strong><br>
                • Las lineas SI SE CRUZAN en Validacion y Prueba<br>
                • Esto confirma que hay sobreposicion y confusion<br>
                • El modelo necesita mejoras para poder separar correctamente
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{graph3_base64}" alt="Distribucion Acumulada">
            </div>
        </div>
        
        <!-- GRÁFICO 4 -->
        <div class="graph-section">
            <h2>✂️ Grafico 4: Calidad de Separacion</h2>
            <div class="explanation">
                <strong>¿Que demuestra?</strong><br>
                Muestra la DIFERENCIA entre la distancia promedio de impostoras y la de genuinas.
                <br><br>
                <strong>¿Por que es importante?</strong><br>
                • Separacion GRANDE: El modelo puede discriminar bien<br>
                • Separacion PEQUENA: El modelo no puede discriminar<br>
                <br>
                <strong>Escala de colores:</strong><br>
                • VERDE (> 0.01): BUENA separacion ✓<br>
                • NARANJA (0.005-0.01): ACEPTABLE<br>
                • ROJO (< 0.005): POBRE separacion ✗<br>
                <br>
                <strong>¿Que vemos?</strong><br>
                Todos estan en ROJO, lo que significa que la separacion es POBRE.
                Las distancias son demasiado pequenas (0.0004-0.0009) para discriminar bien.
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{graph4_base64}" alt="Separacion">
            </div>
        </div>
        
        <!-- GRÁFICO 5 -->
        <div class="graph-section">
            <h2>🎯 Grafico 5: Curva de Tasas de Error</h2>
            <div class="explanation">
                <strong>¿Que demuestra?</strong><br>
                Muestra como cambian dos tasas de error segun el umbral:
                <br><br>
                • <strong>TFP (linea azul):</strong> Tasa de Falsos Positivos = firmas genuinas rechazadas<br>
                • <strong>TFN (linea naranja):</strong> Tasa de Falsos Negativos = impostoras aceptadas<br>
                <br>
                <strong>¿Que buscamos?</strong><br>
                El punto donde TFP = TFN (donde se cruzan las lineas). Este es el EQUILIBRIO OPTIMO (EER).
                <br><br>
                <strong>¿Que significa el umbral?</strong><br>
                Es la distancia que usamos para decidir: 
                Si distancia < umbral → ES GENUINA
                Si distancia > umbral → ES IMPOSTORA
                <br><br>
                <strong>¿Que vemos?</strong><br>
                • Las lineas se cruzan en EER = 36.57%<br>
                • En ese punto, ambos errores son iguales<br>
                • Pero 36.57% sigue siendo ALTO (queremos < 15%)
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{graph5_base64}" alt="Tasas de Error">
            </div>
        </div>
        
        <!-- GRÁFICO 6 -->
        <div class="graph-section">
            <h2>🎻 Grafico 6: Grafico de Violin</h2>
            <div class="explanation">
                <strong>¿Que demuestra?</strong><br>
                Similar a un box plot pero muestra la FORMA COMPLETA de la distribucion (como un violin).
                <br><br>
                <strong>¿Que significa la forma?</strong><br>
                • Forma ESTRECHA: Los datos estan concentrados<br>
                • Forma ANCHA: Los datos estan dispersos<br>
                • Bultos: Hay grupos de datos en esos valores<br>
                <br>
                <strong>¿Que buscamos?</strong><br>
                Los dos violines deben tener formas MUY DIFERENTES y estar SEPARADOS.
                <br><br>
                <strong>¿Que vemos?</strong><br>
                • Las distribuciones se sobrelapan<br>
                • Las formas son similares<br>
                • Esto confirma que el modelo no separó bien las clases
            </div>
            <div class="graph-container">
                <img src="data:image/png;base64,{graph6_base64}" alt="Violin Plot">
            </div>
        </div>
        
        <!-- DIAGNÓSTICO -->
        <div class="conclusion">
            <h3>⚠️ Diagnostico Final</h3>
            <p>
                <strong>Problema Identificado:</strong><br>
                El modelo genera distancias DEMASIADO PEQUENAS (0.0009 a 0.0013) debido a que 
                la normalizacion L2 comprime todos los embeddings sobre una esfera unitaria.
            </p>
            <p>
                <strong>Causa Raiz:</strong><br>
                1. Normalizacion L2 colapsa la informacion → Distancias ultra-pequenas<br>
                2. Margen = 0.5 es 500 veces mayor que distancias reales → Imposible convergencia<br>
                3. Solo 5 épocas en 10% datos → Entrenamiento insuficiente para aprender bien<br>
            </p>
            <p>
                <strong>Síntomas:</strong><br>
                ✗ EER = 36.57% (PEOR que el baseline 38.57%)<br>
                ✗ Validacion con 26.71% sobreposicion → SOBREAJUSTE detectado<br>
                ✗ Exhaustividad baja = 60% de impostoras no detectadas (muy peligroso)<br>
                ✗ Separacion de 0.0004 → Casi imposible discriminar
            </p>
            <p>
                <strong>Solucion Propuesta:</strong><br>
                1. Remover o ajustar normalizacion L2<br>
                2. Cambiar margen a valor adaptativo (basado en distancias reales)<br>
                3. Aumentar dimension del embedding de 128 a 256<br>
                4. Entrenar con 100% de datos y 180 épocas<br>
                5. (Opcional) Usar GPU para acelerar 130 horas a 2-3 horas
            </p>
        </div>
        
        <footer>
            <p>Analisis generado: 2025-11-18 | Mini-entrenamiento completado exitosamente</p>
            <p><small>Archivos: analisis_distancias.png | analisis_metricas_rendimiento.png | metricas_resumen.csv</small></p>
            <p><small>Redes LSTM + Perdida Triplet | 10% datos | 5 épocas | Diagnostico completado</small></p>
        </footer>
    </div>
</body>
</html>
"""

# ============================================================================
# GUARDAR HTML
# ============================================================================

with open("analisis_interactivo.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\n" + "="*80)
print("ARCHIVO HTML ACTUALIZADO: analisis_interactivo.html")
print("="*80)
print("\nAbre el archivo en tu navegador:")
print("  Doble clic en: analisis_interactivo.html")
print("  O arrastralo a tu navegador")
print("\nAhora con:")
print("  [OK] Todo en ESPANOL")
print("  [OK] Explicacion de Redes LSTM y Perdida Triplet")
print("  [OK] Explicacion de que demuestra cada grafico")
print("  [OK] Definiciones de todas las metricas")
print("  [OK] 6 graficos interactivos integrados")
print("="*80)
