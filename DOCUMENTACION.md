# 📋 DOCUMENTACIÓN COMPLETA - RECONOCIMIENTO DE FIRMAS

## 🎯 RESUMEN EJECUTIVO

**Proyecto**: Verificar si una firma es genuina o falsificada usando Deep Learning  
**Tecnología**: LSTM (Long Short-Term Memory) + Triplet Loss  
**Meta**: Acertar en >85% de casos (EER < 15%)  
**Estado**: Entrenamiento en progreso  
**Datos**: 1,600 firmas de 40 personas (limpio, sin errores)

---

## 📊 SECCIÓN 1: LOS DATOS

### 1.1 ¿Qué Datos Tenemos?

Cada firma registrada electrónicamente contiene:
- **208 puntos** en secuencia temporal (cómo se escribió)
- **4 valores por punto**:
  - X: coordenada horizontal
  - Y: coordenada vertical
  - T: tiempo desde inicio
  - P: presión del bolígrafo

**Total**: 1,600 firmas × 208 puntos × 4 valores = **1,331,200 datos**

### 1.2 Archivo Principal: X_features.npy

```
Archivo: X_features.npy
Tamaño: 5.08 MB
Estructura: Matriz de 1600 × 208 × 4
Ubicación: Task2_Preprocesado/
Estado: ✓ VERIFICADO (0 NaN, 0 Inf, normalizado)
```

**Ejemplo visual de una firma:**
```
Punto 1: (X=45.2, Y=120.3, T=0.01s, P=0.8)
Punto 2: (X=46.1, Y=119.8, T=0.02s, P=0.85)
...
Punto 208: (X=200.5, Y=150.2, T=2.45s, P=0.3)
```

### 1.3 Normalización: Z-Score

**¿Por qué normalizar?**
- X puede variar 0-1000, Y puede variar 0-500, T puede variar 0-5 segundos
- El modelo se confunde con escalas diferentes
- Solución: Convertir todo a escala estándar (-4 a +3)

**Fórmula**:
```
valor_normalizado = (valor - media) / desviación_estándar
```

**Parámetros usados** (guardados en normalization_params.json):
```json
{
  "shape": [208, 4],
  "mean": [0.1291701943, -0.0409635082, -0.0173108261, 0.0276430771],
  "std": [0.8835738897, 0.9152904153, 0.9258862734, 0.9039105177]
}
```

**Verificación**: ✓ Exactamente calculados desde X_features.npy

### 1.4 División de Datos

```
1600 firmas totales
├── 1024 (64%) = Entrenamiento (modelo aprende)
├── 256 (16%) = Validación (verifica aprendizaje)
└── 320 (20%) = Test (mide resultado final - EER)
```

**Archivos de soporte:**
- **Y_user.npy**: Etiqueta de usuario (1-40) para cada firma
- **M_mask.npy**: Máscara de padding (dónde terminan datos válidos)

### 1.5 Limpieza de Datos

```
Verificación realizada:
✓ 0 valores NaN (no hay datos faltantes)
✓ 0 valores Inf (no hay infinitos)
✓ Rango verificado: [-4.02, 3.39] (normal para Z-Score)
✓ 100% de datos válidos
```

---

## 🧠 SECCIÓN 2: ARQUITECTURA LSTM

### 2.1 ¿Qué es LSTM?

**LSTM = Long Short-Term Memory** (Memoria a Largo Plazo - Corto Plazo)

Es una red neuronal especial que:
- Procesa datos **secuenciales** (punto por punto de la firma)
- **Recuerda** información importante mientras procesa
- Puede olvidar información irrelevante
- Detecta patrones en el tiempo

**Analogía**: Como un lector humano que:
- Lee palabra por palabra
- Recuerda el contexto importante
- Olvida detalles sin importancia
- Entiende el significado completo

### 2.2 Arquitectura Completa

```
ENTRADA (208 puntos × 4 features)
    ↓
[Masking Layer] → Ignora puntos con padding
    ↓
[LSTM_1: 128 unidades] → Aprende patrones locales
    ├─ return_sequences=True (pasa todos los 208 outputs)
    ├─ dropout=0.3 (30% de neuronas se ignoran aleatoriamente)
    └─ Parámetros: ~75,008
    ↓
[LSTM_2: 64 unidades] → Aprende patrones globales
    ├─ return_sequences=False (pasa solo el último output)
    ├─ dropout=0.3
    └─ Parámetros: ~36,608
    ↓
[Dense_1: 256 unidades] → Transformación lineal
    ├─ Activación: ReLU (retifica negativos a 0)
    ├─ L2 Regularization (penaliza pesos grandes)
    └─ Dropout=0.3
    ↓
[Dense_2: 128 unidades] → Antes de embedding
    ├─ Activación: ReLU
    ├─ L2 Regularization
    └─ Parámetros: ~33,024
    ↓
[L2 Normalization] → Normaliza a longitud 1
    ↓
SALIDA: Embedding de 128 dimensiones (el "código" de la firma)
```

### 2.3 Conteo de Parámetros

```
Total parámetros: 167,040
├─ LSTM_1: 75,008 (44.9%)
├─ LSTM_2: 36,608 (21.9%)
├─ Dense_1: 33,024 (19.8%)
├─ Dense_2: 16,512 (9.9%)
└─ Biases: ~5,888 (3.5%)

Tamaño en memoria: ~652 KB
Después de compresión: ~702 KB
```

### 2.4 Dropout: Regularización

**¿Qué es dropout?**
- Apagar aleatoriamente el 30% de neuronas durante entrenamiento
- Obliga a la red a aprender de forma robusta
- Previene overfitting (memorizar datos)

**Analogía**: Como entrenar un equipo deportivo:
- Si cada día juegan los mismos 11, se vuelven predecibles
- Si rotas jugadores, cada uno debe jugar independientemente
- Resultado: equipo más flexible y robusto

---

## 🎓 SECCIÓN 3: TRIPLET LOSS (Cómo Aprende)

### 3.1 Concepto Fundamental

**Triplet** = Conjunto de 3 ejemplos:
1. **Anchor (A)**: Firma de referencia (ej: Juan firma 1)
2. **Positive (P)**: Otra firma del mismo usuario (ej: Juan firma 5)
3. **Negative (N)**: Firma de usuario diferente (ej: Pedro firma 8)

### 3.2 Objetivo del Triplet Loss

```
Objetivo: Aprender embedding tal que:
├─ distancia(A, P) < distancia(A, N)  [Genuinas más cerca que falsas]
└─ Con margen de seguridad m

Matemáticamente:
distancia(A, P) + m < distancia(A, N)

Ejemplo con m=0.5:
dist(A,P)=0.3, dist(A,N)=1.2
0.3 + 0.5 = 0.8 < 1.2 ✓ Correcto
```

### 3.3 Visualización del Espacio de Embeddings

```
Sin entrenamiento (random):
┌─────────────────────────────┐
│ 🔴 🔵 🔴 🔵 🔴 🔵      │
│ 🔵 🔴 🔵 🔴 🔵 🔴      │
│ Caos - no hay separación    │
└─────────────────────────────┘

Con Triplet Loss (antes de mejorar - margin=0.1):
┌─────────────────────────────┐
│ 🔴🔴 🔵🔵 🔴🔴 🔵🔵  │
│  🔵  🔴  🔵  🔴       │
│ Poco separados - solapan    │
└─────────────────────────────┘

Con Triplet Loss mejorado (margin=0.5):
┌─────────────────────────────┐
│ 🔴🔴🔴      🔵🔵🔵     │
│                            │
│                            │
│         [MARGEN]           │
│                            │
│ Muy separados - claro      │
└─────────────────────────────┘
```

### 3.4 Parámetro Crítico: MARGIN

**Margin** = Brecha mínima de seguridad entre genuinas y falsas

```
Margin = 0.1 (TOO SMALL - actual problema):
├─ dist(genuina) = 0.4
├─ dist(falsa) = 0.5
├─ Diferencia = 0.1 (muy pequeña, se confunde)
└─ Resultado: EER = 40.29% ❌

Margin = 0.5 (MEJORADO - solución):
├─ dist(genuina) = 0.3
├─ dist(falsa) = 1.2
├─ Diferencia = 0.9 (amplia, clara distinción)
└─ Resultado esperado: EER = 10-15% ✓
```

### 3.5 Hard Triplet Mining (Minería de Triples Duros)

**¿Qué es?**
- Seleccionar triples que son más difíciles de clasificar
- No usar triples obvios (demasiado fáciles)

**Ejemplo:**
```
Tripple Fácil:
- A: Juan firma normal
- P: Juan firma normal (muy parecida)
- N: Pedro firma completamente diferente
→ Muy fácil de distinguir

Tripple Duro:
- A: Juan firma rápida
- P: Juan firma lenta (estilo diferente)
- N: Falsificación muy buena de Juan
→ Muy difícil de distinguir (enseña mejor)

Usar triples duros = Modelo aprende mejor
```

---

## 📈 SECCIÓN 4: ENTRENAMIENTO

### 4.1 Configuración Mejorada (train_proper.py)

```python
CONFIGURACIÓN ACTUAL:
├─ Epochs: 180 (iteraciones completas del dataset)
├─ Batch Size: 64 (64 triples simultáneamente)
├─ Learning Rate: 0.01 (velocidad de aprendizaje)
├─ LR Decay: steps=450, rate=0.94 (reduce LR cada 450 iteraciones)
├─ Margin: 0.5 (separación mínima)
├─ Hard Negatives: 4 (minería de triples duros)
├─ Optimizer: Adam (ajusta parámetros inteligentemente)
└─ Loss: Euclidean Triplet Loss
```

### 4.2 Proceso Típico de Entrenamiento

```
Epoch 1/180
├─ Generar 500 triples (A, P, N)
├─ Forward pass (modelo predice embeddings)
├─ Calcular loss (qué tan malo es)
├─ Backward pass (calcular gradientes)
├─ Actualizar pesos (ajustar modelo)
└─ Loss: 0.120 (empieza alto)

Epoch 90/180 (medio del entrenamiento)
└─ Loss: 0.065 (bajando, modelo aprende)

Epoch 180/180 (final)
└─ Loss: 0.045 (convergió, modelo listo)
```

### 4.3 Métricas Monitoreadas

```
Durante entrenamiento:
├─ Loss: Distancia euclidiana entre embeddings
├─ Val_Loss: Pérdida en conjunto de validación
└─ Learning Rate: Se reduce gradualmente

Al final (en test):
├─ EER: Tasa de error equilibrada (META: < 15%)
├─ Threshold óptimo: Valor de distancia para decisión
├─ Embeddings guardados: Para análisis posterior
└─ Separación de clases: Qué tan bien separadas están
```

---

## ✅ SECCIÓN 5: EVALUACIÓN (EER)

### 5.1 ¿Qué es EER?

**EER = Equal Error Rate** (Tasa de Error Equilibrada)

En el test:
1. Calcular embeddings de todas las firmas
2. Calcular distancias entre pares (A, B)
3. Determinar threshold óptimo donde:
   - False Reject Rate (FRR) = False Accept Rate (FAR)
   - Es decir, errores positivos = errores negativos

### 5.2 Interpretación de EER

```
EER 5%  = Excelente (1 error cada 20 decisiones)
EER 10% = Muy bueno (1 error cada 10 decisiones)
EER 15% = Bueno (1 error cada 6-7 decisiones) ← NUESTRA META
EER 20% = Aceptable (1 error cada 5 decisiones)
EER 40% = Pobre (2 errores cada 5 decisiones) ← ESTADO ACTUAL
EER 50% = Random (sin información, al azar)
```

### 5.3 Curva ROC y Threshold

```
                True Positive Rate
                      ↑
                 100% ╱╲
                      ╱  ╲
                     ╱    ╲
                    ╱      ╲
                   ╱        ╲
              50% ├──────────╲──
                  ╱          ╲
                 ╱            ╲
                ╱              ╲
             0%└─────────────────╲─→ False Positive Rate
                0%  50%  100%   

El punto donde FPR = FNR es el EER
(línea diagonal en gráficos típicos)
```

### 5.4 Archivo de Resultados: lstm_stacked_final_results.json

```json
{
  "eer_percent": 40.291,        // Tasa error actual
  "optimal_threshold": 0.490,   // Distancia de decisión
  "embedding_dim": 128,         // Dimensionalidad del código
  "train_samples": 1024,        // Firmas usadas para entrenar
  "val_samples": 256,           // Firmas para validación
  "test_samples": 320,          // Firmas para test
  "mean_intra_dist": 0.442,     // Distancia promedio genuinas
  "mean_inter_dist": 0.562,     // Distancia promedio falsas
  "separation": 0.120           // (inter - intra) = calidad de separación
}
```

---

## 🔧 SECCIÓN 6: ARCHIVOS DEL PROYECTO

### 6.1 Estructura de Carpetas

```
c:\Users\user\Downloads\Dataset\
│
├── 📂 Task2_Preprocesado/          [Datos limpios]
│   ├── X_features.npy              (5.08 MB) - Firmas
│   ├── Y_user.npy                  (0.01 MB) - Etiquetas
│   └── M_mask.npy                  (4.88 MB) - Máscaras
│
├── 📄 normalization_params.json     (1.5 KB)  ✓ VALIDADO
├── 📄 DOCUMENTACION.md              (Este archivo)
├── 📄 PROGRESO_ACTUAL.md            (Progreso y debugging)
│
├── 🐍 train_proper.py              (Script entrenamiento actual)
├── 🐍 signature_training.py         (Módulo de utilidades)
├── 🐍 train_final.py               (Alternativa backup)
│
└── 📦 embedding_network_final.h5    (Modelo - se genera al entrenar)
    🎯 lstm_stacked_final_results.json (Resultados - se genera al entrenar)
    🎯 lstm_stacked_final_SUCCESS.npz  (Embeddings test - se genera al entrenar)
```

### 6.2 Archivos Críticos en Detalle

| Archivo | Tipo | Tamaño | Propósito | Status |
|---------|------|--------|-----------|--------|
| X_features.npy | Datos | 5.08 MB | 1600 firmas (208×4) | ✓ Limpio |
| Y_user.npy | Etiquetas | 12 KB | Usuario (1-40) | ✓ Verificado |
| M_mask.npy | Máscara | 4.88 MB | Padding info | ✓ Verificado |
| normalization_params.json | Config | 1.5 KB | μ, σ para Z-Score | ✓ VALIDADO |
| train_proper.py | Script | ~15 KB | Entrenamiento mejorado | ✓ Listo |
| signature_training.py | Módulo | ~8 KB | Clases auxiliares | ✓ Importable |
| embedding_network_final.h5 | Modelo | ~702 KB | Red entrenada | ⏳ Generará |
| lstm_stacked_final_results.json | Métricas | ~540 B | EER y threshold | ⏳ Generará |

---

## 🚀 SECCIÓN 7: FLUJO COMPLETO (End-to-End)

### 7.1 Fase 1: Preparación

```
1. Cargar X_features.npy
   └─ Verificar: shape (1600, 208, 4), sin NaN, sin Inf

2. Cargar normalization_params.json
   └─ Extraer μ (mean) y σ (std) para 4 features

3. Crear splits:
   └─ Train: 1024, Val: 256, Test: 320 (basado en Y_user)

4. Crear arquitectura LSTM
   └─ Input→Mask→LSTM→Dense→L2Norm→Output(128-dim)
```

### 7.2 Fase 2: Entrenamiento

```
1. Para cada época (1-180):
   ├─ Generar 500 triples (A, P, N) del training set
   ├─ Forward pass: (A, P, N) → embeddings (3 × 128)
   ├─ Calcular Triplet Loss con margin=0.5
   ├─ Backward pass: calcular gradientes
   ├─ Actualizar pesos: w = w - lr × ∇loss
   ├─ Validar en validation set
   └─ Mostrar: "Epoch X/180 - loss: Y.ZZZ"

2. Decaimiento de learning rate:
   ├─ Cada 450 iteraciones
   └─ lr = lr × 0.94 (reduce 6% cada paso)
```

### 7.3 Fase 3: Evaluación (Test)

```
1. Cargar modelo guardado: embedding_network_final.h5

2. Calcular embeddings de test set:
   └─ 320 firmas → 320 embeddings (128-dim cada uno)

3. Calcular matriz de distancias:
   └─ Distancia euclidiana entre todos los pares

4. Encontrar threshold óptimo:
   ├─ Para cada threshold posible
   ├─ Calcular FRR (genuinas rechazadas)
   ├─ Calcular FAR (falsas aceptadas)
   ├─ Encontrar punto donde FRR = FAR
   └─ EER = FRR en ese punto

5. Generar reporte:
   └─ Guardar lstm_stacked_final_results.json con EER
```

---

## 💡 SECCIÓN 8: CONCEPTOS CLAVE

### 8.1 Deep Learning Básico

**Red Neuronal**: Sistema de "neuronas" conectadas que aprende patrones

```
Entrada → Pesos → Activación → Output
(X)       (w)     (ReLU)       (Y)

El modelo ajusta los pesos (w) para minimizar error
```

### 8.2 Backpropagation

Algoritmo para calcular cómo cambiar los pesos:
1. Forward: Calcular predicción
2. Calcular error (loss)
3. Backward: Propagar error hacia atrás
4. Calcular gradientes (qué dirección cambiar)
5. Actualizar pesos

**Analogía**: Como un estudiante:
- Intenta resolver problema
- Verifica respuesta (error)
- Analiza dónde falló
- Ajusta estrategia
- Intenta de nuevo

### 8.3 Regularización L2

Penalización adicional a pesos grandes:
```
Loss Total = Triplet Loss + λ × (suma de w²)

Sin L2: modelo puede usar pesos enormes (overfitting)
Con L2: modelo usa pesos moderados (generaliza mejor)

λ = 1e-4 (muy pequeño, es una "multa" suave)
```

### 8.4 Dropout

Apagar 30% neuronas aleatoriamente durante entrenamiento:
- Previene que neuronas coextraigan información
- Obliga a cada neurona a ser independiente
- Resultado: modelo más robusto

### 8.5 Normalización L2 (en Output)

Escalar embedding a longitud 1:
```
embedding_original = [0.5, -0.3, 0.8, ...]
embedding_normalizado = [0.5, -0.3, 0.8, ...] / ||vector||
resultado: vector unitario en 128 dimensiones

Ventaja: Distancia euclidiana = 1 - similitud coseno
→ Mejor interpretable
```

---

## 🎯 SECCIÓN 9: PRÓXIMOS PASOS

### Paso 1: Ejecutar Entrenamiento
```powershell
cd "c:\Users\user\Downloads\Dataset"
python train_proper.py
```

### Paso 2: Monitorear Progreso
- Observar reducción de loss
- Verificar que no hay errores
- Tiempo estimado: 2-3 horas

### Paso 3: Verificar Resultado
```json
Buscar en lstm_stacked_final_results.json:
{
  "eer_percent": ?
}

Si EER < 15% → ✓ ÉXITO
Si EER ≥ 15% → Ajustar hyperparámetros
```

### Paso 4: Preparar Producción
Guardar 3 archivos críticos:
- embedding_network_final.h5
- lstm_stacked_final_results.json
- normalization_params.json

---

**Documento actualizado**: 17 de noviembre de 2025  
**Versión**: 2.0 (Con explicación de error EER y conceptos completos)  
**Estado**: Listo para entrenamiento mejorado
