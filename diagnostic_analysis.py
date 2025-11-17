import numpy as np
import json
from scipy.spatial.distance import pdist, squareform

print('='*80)
print('ANÁLISIS DIAGNÓSTICO DEL PROBLEMA')
print('='*80)

# Cargar embeddings y etiquetas
data = np.load('lstm_stacked_final_SUCCESS.npz')

if 'embeddings_test' in data:
    embeddings = data['embeddings_test']
    Y_test = data['Y_test']
elif 'embeddings' in data:
    embeddings = data['embeddings']
    Y_test = data['Y_test']
else:
    raise KeyError('No se encontraron embeddings en el archivo NPZ.')

print(f'\n[1] Verificar embeddings:')
print(f'  Shape: {embeddings.shape}')
print(f'  dtype: {embeddings.dtype}')
print(f'  Min: {embeddings.min():.6f}')
print(f'  Max: {embeddings.max():.6f}')
print(f'  Mean: {embeddings.mean():.6f}')
print(f'  Std: {embeddings.std():.6f}')
print(f'  NaN: {np.isnan(embeddings).sum()}')
print(f'  Inf: {np.isinf(embeddings).sum()}')

# Verificar L2 normalization (norma debe ser ~1)
norms = np.linalg.norm(embeddings, axis=1)
print(f'\n[2] Norma L2 de embeddings (deben estar normalizados a 1):')
print(f'  Min: {norms.min():.6f}')
print(f'  Max: {norms.max():.6f}')
print(f'  Mean: {norms.mean():.6f}')
print(f'  Std: {norms.std():.6f}')

# Calcular distancias
distances = squareform(pdist(embeddings, metric='euclidean'))

print(f'\n[3] Análisis de distancias:')
print(f'  Min distancia: {distances[distances > 0].min():.6f}')
print(f'  Max distancia: {distances.max():.6f}')
print(f'  Mean distancia: {distances[distances > 0].mean():.6f}')

# Separar distancias genuinas vs impostores
genuine_distances = []
impostor_distances = []

for i in range(len(Y_test)):
    for j in range(i + 1, len(Y_test)):
        if Y_test[i] == Y_test[j]:
            genuine_distances.append(distances[i, j])
        else:
            impostor_distances.append(distances[i, j])

genuine_distances = np.array(genuine_distances)
impostor_distances = np.array(impostor_distances)

print(f'\n[4] Distancias GENUINAS (misma persona):')
print(f'  Count: {len(genuine_distances)}')
print(f'  Min: {genuine_distances.min():.6f}')
print(f'  Max: {genuine_distances.max():.6f}')
print(f'  Mean: {genuine_distances.mean():.6f}')
print(f'  Std: {genuine_distances.std():.6f}')

print(f'\n[5] Distancias IMPOSTORAS (diferentes personas):')
print(f'  Count: {len(impostor_distances)}')
print(f'  Min: {impostor_distances.min():.6f}')
print(f'  Max: {impostor_distances.max():.6f}')
print(f'  Mean: {impostor_distances.mean():.6f}')
print(f'  Std: {impostor_distances.std():.6f}')

# Separación
print(f'\n[6] ANÁLISIS DE SEPARACIÓN:')
overlap = ((genuine_distances.max() > impostor_distances.min()) and 
           (impostor_distances.max() > genuine_distances.min()))
print(f'  ¿Hay overlap entre distribuciones?: {overlap}')

if genuine_distances.mean() < impostor_distances.mean():
    separacion = impostor_distances.mean() - genuine_distances.mean()
    print(f'  ✓ Separación correcta (genuinas < impostoras)')
    print(f'  Diferencia de medias: {separacion:.6f}')
else:
    print(f'  ❌ PROBLEMA: Las distancias están invertidas o mal calibradas')
    print(f'  Media genuinas: {genuine_distances.mean():.6f}')
    print(f'  Media impostoras: {impostor_distances.mean():.6f}')

print(f'\n[7] Percentiles GENUINAS:')
for p in [10, 25, 50, 75, 90]:
    print(f'  P{p:2d}: {np.percentile(genuine_distances, p):.6f}')

print(f'\n[8] Percentiles IMPOSTORAS:')
for p in [10, 25, 50, 75, 90]:
    print(f'  P{p:2d}: {np.percentile(impostor_distances, p):.6f}')

print(f'\n' + '='*80)
print('CONCLUSIÓN:')
print('='*80)

if genuine_distances.mean() < impostor_distances.mean() and separacion > 0.3:
    print('✓ Embeddings bien calibrados, pero threshold/margin puede estar mal')
elif genuine_distances.mean() > impostor_distances.mean():
    print('❌ CRÍTICO: Embeddings INVERTIDOS (genuinas > impostoras)')
    print('Esto significa que el Triplet Loss NO convergió correctamente')
    print('Causa: Margin demasiado pequeño, learning rate, o generación de triplets')
else:
    print('⚠ Separación pobre entre clases')
    print('Causa: Modelo NO aprendió a distinguir')

print('='*80)
