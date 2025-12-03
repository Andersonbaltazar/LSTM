# LSTM Stacked - Reconocimiento de Firmas

## Objetivo

Entrenar una red LSTM con Triplet Loss para verificacion de firmas digitales.
**Metrica:** Equal Error Rate (EER)
**Meta:** EER < 15%

---

## Estructura

```
Dataset/
├── README.md                    ← Este archivo
├── train_proper.py              ← Entrenamiento (180 epocas, 100% datos)
├── test_fixed.py                ← Validacion rapida (10% datos, 5 epocas)
├── analyze_mini.py              ← Analisis de metricas y graficos
├── generar_html_espanol.py      ← Visualizacion HTML interactiva
└── Task2_Preprocesado/          ← DATOS (no tocar)
    ├── X_features.npy           # 1600 firmas x 400 timesteps x 4 features
    ├── Y_user.npy               # Etiquetas de usuario
    └── M_mask.npy               # Mascaras de padding
```

---

## Arquitectura

LSTM Stacked + Triplet Loss

```
Entrada: (400, 4) timesteps
    |
LSTM_1: 128 unidades, dropout=0.3, return_sequences=True
    |
LSTM_2: 64 unidades, dropout=0.3, return_sequences=False
    |
Dense: 256 unidades, ReLU, dropout=0.3
    |
Embedding: 256 dimensiones
    |
Triplet Loss: margin=0.25

Total: 199,936 parametros
```

---

## Parametros

```python
epochs = 180               # Entrenamiento completo
batch_size = 64           # Tamaño de batch
margin = 0.25             # Margen Triplet Loss
embedding_dim = 256       # Dimensiones embedding
learning_rate = 0.01      # Tasa de aprendizaje (con decay)
hard_negatives = 4        # Negativos duros por triplet
data_percentage = 100%    # Usa todas las 1600 firmas
```

---

## Ejecucion

### 1. Entrenamiento Completo (RECOMENDADO)
```powershell
.\.venv\Scripts\python.exe train_proper.py
```
- Tiempo: ~130 horas (CPU) / 6-10 horas (GPU)
- Genera: `embedding_network_final.h5`, `lstm_stacked_final_results.json`

### 2. Validacion Rapida (Opcional - verificar que funciona)
```powershell
.\.venv\Scripts\python.exe test_fixed.py
```
- Tiempo: ~5 minutos
- Genera: `test_fixed_results.json`, `embedding_network_test_fixed.h5`

### 3. Analisis de Resultados (Despues de entrenar)
```powershell
.\.venv\Scripts\python.exe analyze_mini.py
```
- Calcula EER, FAR, FRR
- Genera graficos (PNG)
- Exporta metricas (CSV)

### 4. Visualizacion HTML (Despues de analisis)
```powershell
.\.venv\Scripts\python.exe generar_html_espanol.py
Invoke-Item analisis_interactivo.html
```
- Muestra 6 graficos interactivos
- Explicaciones en espanol

---

## Metricas de Salida

Archivo: `lstm_stacked_final_results.json`

```json
{
  "model_info": {
    "total_parameters": 199936,
    "embedding_dimension": 256,
    "margin": 0.25
  },
  "validation_metrics": {
    "val_loss": "...",
    "val_eer": "..."
  },
  "test_metrics": {
    "test_eer": "← META: < 15%",
    "test_far": "...",
    "test_frr": "...",
    "genuine_mean_distance": "...",
    "impostor_mean_distance": "..."
  }
}
```

---

## Datos de Entrada

Archivo: `Task2_Preprocesado/X_features.npy`

```
Shape: (1600, 400, 4)
- 1600 firmas digitales
- 400 timesteps cada una
- 4 features por timestep: x, y, vx, vy
- Rango: normalizado (Min-Max o Z-score)
- Estado: limpio (sin NaN/Inf)
```

---

## Flujo Completo

```
1. Ejecutar:
   .\.venv\Scripts\python.exe train_proper.py
   ↓ genera embedding_network_final.h5

2. Analizar:
   .\.venv\Scripts\python.exe analyze_mini.py
   ↓ genera graficos PNG

3. Visualizar:
   .\.venv\Scripts\python.exe generar_html_espanol.py
   ↓ abre analisis_interactivo.html

4. Revisar:
   - EER < 15%? → EXITO
   - EER 15-20%? → Bueno, puede iterarse
   - EER > 20%? → Revisar parametros
```

---

## Notas Importantes

- No modificar: Archivos en `Task2_Preprocesado/`
- Parametros: `margin=0.25`, `embedding_dim=256`
- test_fixed.py: Para verificaciones rapidas
- Reproducibilidad: Seed fijo en 42

---

Ultima actualizacion: 2 de Diciembre de 2025
