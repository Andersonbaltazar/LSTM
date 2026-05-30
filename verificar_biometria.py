import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cosine
# Importamos la función original de arquitectura directamente de tu script base
from signature_training import build_encoder 

# --- CONFIGURACIÓN DE PARÁMETROS ---
MODEL_PATH = "embedding_network_mini.h5" 
X_PATH = "Task2_Preprocesado/X_features.npy"
Y_PATH = "Task2_Preprocesado/Y_user.npy"
CONFIDENCE_THRESHOLD = 0.70
EMBEDDING_DIM = 256  # Dimensión real configurada en entrenar_rapido.py

print("=" * 80)
print("             SIMULADOR DE PASO BIOMÉTRICO (MFA STEP-UP)")
print("=" * 80)

# 1. Cargar la arquitectura original e inyectar los pesos
print("\n[1] Cargando artefactos...")
try:
    # Usamos el constructor original de tu archivo signature_training.py
    model = build_encoder(embedding_dim=EMBEDDING_DIM)
    model.load_weights(MODEL_PATH)
    print(f"  ✓ Arquitectura oficial cargada y pesos acoplados desde: {MODEL_PATH}")
except Exception as e:
    print(f"  ❌ Error al inicializar el modelo con sus pesos: {e}")
    exit()

try:
    X = np.load(X_PATH).astype(np.float32)
    Y = np.load(Y_PATH).astype(np.int32)
    print(f"  ✓ Dataset cargado: X={X.shape}, Y={Y.shape}")
except Exception as e:
    print(f"  ❌ Error al cargar los archivos .npy: {e}")
    exit()

# Preprocesamiento idéntico: Padding a 400 timesteps
X_padded = np.zeros((len(X), 400, 4), dtype=np.float32)
X_padded[:, :X.shape[1], :] = X
X = X_padded

# 2. Seleccionar Datos Maestros (Enrolamiento)
unique_users, counts = np.unique(Y, return_counts=True)
valid_users = unique_users[counts >= 6]

if len(valid_users) == 0:
    print("❌ No hay usuarios con suficientes firmas (mínimo 6) para enrolar y testear.")
    exit()

target_user = valid_users[0]
user_indices = np.where(Y == target_user)[0]

print(f"\n[2] Simulando ENROLAMIENTO para el Usuario Maestro ID: {target_user}")
enroll_indices = user_indices[:5]
X_enroll = X[enroll_indices]

embeddings_enroll = model.predict(X_enroll, verbose=0)
print(f"  ✓ Generados 5 embeddings de dimensión {embeddings_enroll.shape[1]}")

# Calcular el MEDOID (Firma central del enrolamiento)
dist_matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        dist_matrix[i, j] = cosine(embeddings_enroll[i], embeddings_enroll[j])

medoid_idx = np.argmin(dist_matrix.sum(axis=1))
master_template = embeddings_enroll[medoid_idx]
print(f"  ✓ Template Maestro generado (Medoid index en enrolamiento: {medoid_idx})")


# 3. Función de Verificación Biométrica
def verificar_firma(firma_input, template_maestro):
    embedding_input = model.predict(np.expand_dims(firma_input, axis=0), verbose=0)[0]
    dist_coseno = cosine(embedding_input, template_maestro)
    
    similitud_coseno = 1.0 - dist_coseno
    confidence = (similitud_coseno + 1.0) / 2.0
    
    es_valido = confidence >= CONFIDENCE_THRESHOLD
    return confidence, es_valido


# 4. EJECUCIÓN DE PRUEBAS
print("\n[3] Ejecutando pruebas de validación cruzada...")

# --- PRUEBA A: UNA FIRMA GENUINA NUEVA ---
idx_genuino = user_indices[5]
firma_genuina = X[idx_genuino]
conf_genuina, resultado_genuino = verificar_firma(firma_genuina, master_template)

print("-" * 65)
print("TEST A: Intentando Login con una firma GENUINA")
print(f"  -> Confianza calculada: {conf_genuina:.4f}")
print(f"  -> Umbral requerido:    {CONFIDENCE_THRESHOLD}")
print(f"  -> RESULTADO: {'🟢 ACCESO PERMITIDO (ARC 1.0)' if resultado_genuino else '🔴 ACCESO DENEGADO'}")
print("-" * 65)

# --- PRUEBA B: UNA FIRMA IMPOSTORA (FALSA) ---
impostor_user = np.random.choice([u for u in unique_users if u != target_user])
idx_impostor = np.where(Y == impostor_user)[0][0]
firma_impostora = X[idx_impostor]
conf_impostora, resultado_impostor = verificar_firma(firma_impostora, master_template)

print("TEST B: Intentando Login con una firma IMPOSTORA (Ataque)")
print(f"  -> Usuario atacante ID: {impostor_user}")
print(f"  -> Confianza calculada: {conf_impostora:.4f}")
print(f"  -> Umbral requerido:    {CONFIDENCE_THRESHOLD}")
print(f"  -> RESULTADO: {'🟢 ACCESO PERMITIDO ⚠️ ERROR!' if resultado_impostor else '🔴 ACCESO DENEGADO (Sistema Seguro)'}")
print("-" * 65)