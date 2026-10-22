# FundiVision - demo de inspeccion en vivo
#
# 1. Carga el pipeline que entreno el notebook (modelos/fundivision.joblib)
# 2. Toma fotos de piezas (por defecto el conjunto test, que el modelo nunca vio)
# 3. Muestra cada foto con el veredicto OK / DEFECTO y el tiempo que tardo
#
# Teclas: ESPACIO = siguiente pieza, ESC = salir
# Uso:    python demo_fundivision.py                 (fotos del conjunto test)
#         python demo_fundivision.py foto.jpeg       (una foto)
#         python demo_fundivision.py carpeta/        (todas las fotos de una carpeta)

import os
import sys
import time
import random
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

RUTA_MODELO = os.path.join("modelos", "fundivision.joblib")
RUTA_DATOS = "data"
extensiones = ('.jpg', '.jpeg', '.png', '.bmp')

# 1. Cargar el pipeline entrenado
if not os.path.exists(RUTA_MODELO):
    print("No encuentro", RUTA_MODELO, "- ejecute primero el notebook completo para entrenar y guardar el modelo.")
    sys.exit(1)
print("Cargando modelo...")
pipeline = joblib.load(RUTA_MODELO)
modelo = pipeline['modelo']
descriptor = pipeline['descriptor']
TAMANO = pipeline['tamano']
CLASES = pipeline['clases']
print("Modelo listo:", descriptor, "+", type(modelo).__name__)

# 2. Mismos descriptores del notebook (deben ser identicos a los del entrenamiento)
def extraer_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

def extraer_lbp(img, P=8, R=1):
    lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float32")
    return hist / (hist.sum() + 1e-7)

def extraer_glcm(img):
    glcm = graycomatrix((img // 4).astype(np.uint8), distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)
    vector = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        vector.extend(graycoprops(glcm, prop).ravel())
    return np.array(vector)

def inspeccionar(ruta):
    t0 = time.time()
    gris = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(gris, TAMANO, interpolation=cv2.INTER_AREA)
    if descriptor == 'HOG':
        X = pipeline['escaladores']['HOG'].transform([extraer_hog(img)])
    elif descriptor == 'LBP':
        X = pipeline['escaladores']['LBP'].transform([extraer_lbp(img)])
    elif descriptor == 'GLCM':
        X = pipeline['escaladores']['GLCM'].transform([extraer_glcm(img)])
    else:
        F = np.hstack([extraer_hog(img), extraer_lbp(img), extraer_glcm(img)])
        X = pipeline['scaler_fusion'].transform([F])
        if descriptor == 'Fusión + PCA':
            X = pipeline['pca'].transform(X)
    pred = CLASES[int(modelo.predict(X)[0])]
    return pred, 1000 * (time.time() - t0), gris

# 3. Lista de fotos a inspeccionar
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    fotos = [sys.argv[1]]
elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    fotos = [os.path.join(sys.argv[1], f) for f in sorted(os.listdir(sys.argv[1])) if f.lower().endswith(extensiones)]
else:
    fotos = []
    for actual, _, archivos in os.walk(RUTA_DATOS):
        if os.path.basename(os.path.dirname(actual)) == 'test':
            fotos += [os.path.join(actual, f) for f in archivos if f.lower().endswith(extensiones)]
    random.shuffle(fotos)
if not fotos:
    print("No hay fotos para inspeccionar.")
    sys.exit(1)

# 4. Ciclo de inspeccion
inspeccionar(fotos[0])  # calentamiento (la primera llamada siempre es mas lenta)
aciertos, total = 0, 0
for ruta in fotos:
    pred, ms, gris = inspeccionar(ruta)
    real = 'DEFECTO' if 'def_front' in ruta else ('OK' if 'ok_front' in ruta else None)

    frame = cv2.cvtColor(cv2.resize(gris, (600, 600)), cv2.COLOR_GRAY2BGR)
    color = (60, 160, 60) if pred == 'OK' else (40, 60, 220)   # BGR: verde / rojo
    cv2.rectangle(frame, (0, 0), (599, 599), color, 12)
    panel = np.full((150, 600, 3), (39, 33, 28), np.uint8)
    cv2.putText(panel, pred, (20, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 3)
    cv2.putText(panel, f"{ms:.0f} ms", (440, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    if real is not None:
        total += 1
        aciertos += (pred == real)
        cv2.putText(panel, f"real: {real}   aciertos: {aciertos}/{total}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    else:
        cv2.putText(panel, os.path.basename(ruta), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    print(f"{os.path.basename(ruta):25s} -> {pred:8s} ({ms:.0f} ms)" + (f"  real: {real}" if real else ""))

    cv2.imshow("FundiVision - inspeccion (ESPACIO: siguiente, ESC: salir)", np.vstack([panel, frame]))
    t = cv2.waitKey(0)
    if t == 27:
        break

cv2.destroyAllWindows()
if total:
    print(f"\nAciertos: {aciertos} de {total} piezas ({100 * aciertos / total:.1f} %)")
