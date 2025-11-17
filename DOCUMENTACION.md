# 📋 RECONOCIMIENTO DE FIRMAS

## 🎯 ¿Qué es el proyecto?

Verificar si una firma es genuina o falsificada usando inteligencia artificial.

- **Datos**: 1,600 firmas de 40 personas
- **Objetivo**: Diferenciar firmas reales de falsas
- **Meta**: Acertar en >85% de casos (EER < 15%)

---

## 📊 LOS DATOS

**¿Qué tenemos?**
- Cada firma: 208 puntos registrados
- Cada punto: coordenada X, Y, tiempo, presión (4 valores)
- Total: 1,600 firmas × 208 puntos × 4 valores = datos limpios (sin errores)

**Cómo se normalizaron:**
Todos los valores se escalan a una distribución estándar (-4 a +3), como convertir de kilómetros a metros. Así el modelo aprende mejor.

**División:**
- 64% entrenamiento (aprende)
- 16% validación (verifica que aprende bien)
- 20% test (mide el resultado final)

---

## 🧠 EL MODELO

**Concepto simple**: Red neuronal que aprende el "patrón" de cada firma.

```
Firma (208 puntos)
    ↓
Red LSTM (128 → 64 unidades)
    ↓
Capas Densas (256 → 128 unidades)
    ↓
"ID" único de 128 números (embedding)
```

**¿Qué es LSTM?** Es una red que recuerda información importante mientras procesa la firma punto por punto.

**¿Qué es el embedding?** Un "código" único que representa la firma: como una huella digital digital.

---

## 🎓 CÓMO APRENDE: TRIPLET LOSS

**Concepto: Aprender con triples de ejemplos**

Imagina que le muestras al modelo 3 firmas:

```
Firma A (Juan)        ← La REFERENCIA (Anchor)
Firma B (Juan)        ← Otra de Juan (Positivo - debe ser SIMILAR a A)
Firma C (Pedro)       ← De otro usuario (Negativo - debe ser DIFERENTE de A)
```

**¿Qué hace el modelo?**

El modelo convierte cada firma en un "código" de 128 números (embedding). Luego:

1. Calcula **distancia A-B** (debe ser PEQUEÑA = firmas similares)
2. Calcula **distancia A-C** (debe ser GRANDE = firmas diferentes)
3. Si distancia(A,B) < distancia(A,C) → ✅ Correcto
4. Si distancia(A,B) ≥ distancia(A,C) → ❌ Error (el modelo se corrige)

**Analogía simple:**
Como enseñar a un niño: "Esta es mi voz, esta otra también es mi voz (parecida), esta es la voz de otro (diferente)". El niño aprende a reconocer.

**Triplet Loss**: La fórmula matemática que mide qué tan bien lo está haciendo.

**Proceso de entrenamiento:**
- 100 épocas (iteraciones)
- 500 triples por época (500 ejemplos por iteración)
- Cada triple tiene 3 firmas (A, B, C)
- El modelo ajusta sus números internos para mejorar
- Al final, aprende a diferenciar genuinas de falsas

**Resultado**: El modelo aprende a reconocer firmas genuinas vs falsas, como un experto que memoriza patrones.

---

## 📈 ENTRENAMIENTO

**Configuración:**
- 100 épocas (iteraciones)
- 500 triplets por época (ejemplos por iteración)
- Adam optimizer (algoritmo de aprendizaje)
- Decay exponencial (aprende rápido al inicio, lento al final)

**Salida por épocas:**
```
Epoch  10/100 - loss: 0.078 (bajando = aprendiendo ✓)
Epoch  20/100 - loss: 0.072 (sigue bajando ✓)
Epoch 100/100 - loss: 0.050 (convergió ✓)
```

---

## ✅ EVALUACIÓN FINAL

**Métrica: EER (tasa de error equilibrada)**

```
¿Qué tan bien diferencia firmas genuinas de falsas?

EER bajo (5%) = Muy bueno (pocas falsas se aceptan, pocas genuinas se rechazan)
EER alto (40%) = Muy malo (confunde mucho)

Baseline actual: 38.57%
Meta: < 15%
```

**Resultado esperado:**
```json
{
  "eer": 12.5%,
  "mejora": 26% (respecto a baseline)
}
```

---

## 💾 ARCHIVOS

| Archivo | Función |
|---------|---------|
| `X_features.npy` | Datos de las firmas |
| `embedding_network_final.h5` | Modelo entrenado (para usar) |
| `lstm_stacked_final_results.json` | Métricas finales |
| `train_final_model.py` | Script del entrenamiento |

---

## 🚀 EN PRODUCCIÓN

**Cómo verificar una firma nueva:**

```
1. Usuario firma en pantalla/tablet
2. Se procesa (normaliza a formato estándar)
3. Red genera "código" (embedding)
4. Se compara con "código" guardado de referencia
5. Si códigos son similares → GENUINA ✓
   Si códigos son diferentes → FALSA ✗
```

---

**Estado**: Entrenamiento en progreso  
**Objetivo**: EER < 15%  
**Fecha**: 16 de noviembre de 2025
