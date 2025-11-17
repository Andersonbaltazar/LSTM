import json
import numpy as np

print('='*80)
print('VERIFICACIÓN DE PARÁMETROS DE NORMALIZACIÓN')
print('='*80)

# Cargar parámetros guardados
with open('normalization_params.json', 'r') as f:
    saved_params = json.load(f)

print('\n[ARCHIVO] normalization_params.json:')
for i, (m, s) in enumerate(zip(saved_params['mean'], saved_params['std'])):
    print(f'  Feature {i}: mean={m:.10f}, std={s:.10f}')

# Recalcular desde datos
X = np.load('X_features.npy')
X_reshaped = X.reshape(-1, 4)
mean_calc = np.mean(X_reshaped, axis=0)
std_calc = np.std(X_reshaped, axis=0)

print('\n[CÁLCULO] Desde X_features.npy:')
for i, (m, s) in enumerate(zip(mean_calc, std_calc)):
    print(f'  Feature {i}: mean={m:.10f}, std={s:.10f}')

# Validar coincidencia
print('\n[VALIDACIÓN]:')
mean_match = np.allclose(saved_params['mean'], mean_calc, rtol=1e-5)
std_match = np.allclose(saved_params['std'], std_calc, rtol=1e-5)
print(f'  ✓ Media coincide: {mean_match}')
print(f'  ✓ Std coincide: {std_match}')

if mean_match and std_match:
    print('\n✓✓✓ PARÁMETROS DE NORMALIZACIÓN VERIFICADOS CORRECTAMENTE')
else:
    print('\n❌ ERROR: Parámetros no coinciden')

# Ver resultados del entrenamiento
print('\n' + '='*80)
print('RESULTADOS DEL ENTRENAMIENTO')
print('='*80)

with open('lstm_stacked_final_results.json', 'r') as f:
    results = json.load(f)

print(f'\n  EER Final: {results["eer_percent"]:.2f}%')
print(f'  Threshold Óptimo: {results["optimal_threshold"]:.4f}')
print(f'  Train Loss Final: {results["final_train_loss"]:.6f}')
print(f'  Val Loss Final: {results["final_val_loss"]:.6f}')
print(f'  Parámetros modelo: {results["model_params"]:,}')
print(f'  Embedding dimensión: {results["embedding_dim"]}')
print(f'  Muestras test: {results["test_samples"]}')
print(f'  Pares genuinos: {results["genuine_samples"]}')
print(f'  Pares impostores: {results["impostor_samples"]}')

print('\n' + '='*80)
