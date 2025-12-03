# MLOps - Plan de Entrenamiento

## 1. Conceptos Clave

### Triplet Loss
**Qué es:** Una función de pérdida que entrena el modelo a través de 3 muestras:
- **Anchor:** Una firma de referencia
- **Positiva:** Otra firma del MISMO usuario
- **Negativa:** Una firma de OTRO usuario

**Objetivo:** Que la distancia entre Anchor-Positiva sea MENOR que Anchor-Negativa
```
Distancia(Anchor, Positiva) < Distancia(Anchor, Negativa) - margin
```

**Margen:** Espacio de seguridad entre clases (aquí = 0.25)

---

### Embedding
**Qué es:** Conversión de una firma (400 timesteps × 4 features) en un vector denso de 256 números.

**Por qué:** El modelo aprende a representar firmas similares cerca en el espacio vectorial.
```
Firma → LSTM → Dense → Embedding (256 dimensiones)
         ↓↓↓    ↓↓↓     ↓↓↓
      Procesa  Comprime Separa en
      secuencia datos    espacio vectorial
```

---

### Equal Error Rate (EER)
**Qué es:** Métrica donde FAR (falsos aceptados) = FRR (falsos rechazados)

**Por qué importa:** Punto de equilibrio del sistema
- Si EER = 10% → Sistema tiene 10% de error en ambos lados
- Meta: EER < 15%

---

### LSTM (Long Short-Term Memory)
**Qué es:** Red neuronal que procesa secuencias (firmas con 400 puntos en el tiempo)

**Por qué:** Recuerda patrones importantes y olvida ruido
```
Entrada (400 timesteps) → LSTM → Extrae características clave
```

---

## 2. Arquitectura del Modelo

```
INPUT: Firma (400 timesteps × 4 features)
   ↓
LSTM_1 (128 unidades) → Extrae patrones principales
   ↓
LSTM_2 (64 unidades) → Refina patrones
   ↓
Dense (256 unidades, ReLU) → Transforma a espacio denso
   ↓
Embedding (256 dimensiones) → Representación final
   ↓
Triplet Loss (margin=0.25) → Entrena separación

Parámetros: 199,936
```

---

## 3. Parámetros de Entrenamiento

```python
epochs = 180              # Iteraciones sobre todo el dataset
batch_size = 64          # Muestras por batch
margin = 0.25            # Espacio entre clases
embedding_dim = 256      # Dimensiones del embedding
learning_rate = 0.01     # Velocidad de aprendizaje
hard_negatives = 4       # Negativos difíciles por triplet
```

---

## 4. Pasos para Entrenar en Data Center

### PASO 1: Preparar Ambiente
```powershell
# En el Data Center:
cd /ruta/del/dataset
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install tensorflow numpy scipy scikit-learn tqdm
```

### PASO 2: Copiar Datos
```
Copiar archivos:
- X_features.npy (5.08 MB) → /dataset/Task2_Preprocesado/
- Y_user.npy (12.6 KB) → /dataset/Task2_Preprocesado/
- M_mask.npy (4.88 MB) → /dataset/Task2_Preprocesado/
```

### PASO 3: Copiar Código
```
Copiar archivos Python:
- train_proper.py
- signature_training.py
- Cualquier archivo .py necesario
```

### PASO 4: Ejecutar Entrenamiento
```powershell
python train_proper.py
```

Esto generará:
- `embedding_network_final.h5` (modelo entrenado)
- `lstm_stacked_final_results.json` (métricas)

### PASO 5: Analizar Resultados
```powershell
python analyze_mini.py
python generar_html_espanol.py
# Abrir: analisis_interactivo.html
```

---

## 5. Monitoreo Durante Entrenamiento

**Salida esperada en consola:**
```
Epoch 1/180
[████████████████████] - Pérdida: 0.45, Val EER: 35.2%
Epoch 2/180
[████████████████████] - Pérdida: 0.42, Val EER: 32.8%
...
Epoch 180/180
[████████████████████] - Pérdida: 0.12, Val EER: 12.3%
```

**Meta:** Val EER < 15%

---

## 6. Flujo Completo

```
1. PREPARACIÓN
   └─ Ambiente Python + Dependencias

2. DATOS
   └─ Copiar Task2_Preprocesado/ (3 archivos .npy)

3. CÓDIGO
   └─ Copiar scripts Python

4. ENTRENAMIENTO
   └─ python train_proper.py (~130 horas en CPU, 6-10 en GPU)

5. EVALUACIÓN
   └─ python analyze_mini.py → Ver métricas

6. VISUALIZACIÓN
   └─ python generar_html_espanol.py → Ver gráficos
```

---

**Nota:** Si Data Center tiene GPU, tiempo se reduce a 6-10 horas.
