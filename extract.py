
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
import xlsxwriter

OUTPUTS_BASE = r"C:\Users\karca\Documents\Decimo_semestre\Vision_Artificial\QUIZ 2\ROIS_AUG" 
EXCEL_NAME   = "caracts_ROIS_aug.xlsx"

CLASSES = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [str(d) for d in range(10)]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

# ROI normalizada (tamaño de patch para features basados en imagen)
NORM_W, NORM_H = 128, 160
FOURIER_K = 12   # nº coeficientes de Fourier (magnitudes) a conservar
HOG_BINS  = 9    # nº de bins para HOG (0-180°)

# =============================================================
# UTILIDADES
# =============================================================

def list_augmented_images(outputs_base):
    """
    Recorre outputs_base y retorna lista de tuplas (class_label, image_path)
    tomando imágenes dentro de "<CLASE>_augmented" y excluyendo "original/".
    """
    pairs = []
    base = Path(outputs_base)
    if not base.exists():
        print(f"⚠️ No existe la carpeta de salida: {outputs_base}")
        return pairs

    for cls in CLASSES:
        folder = base / f"{cls}_augmented"
        if not folder.exists():
            # No todas las clases deben existir; continuamos.
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.parent.name.lower() != "original":
                pairs.append((cls, str(p)))
    print(f"📂 Encontradas {len(pairs)} imágenes aumentadas en '{outputs_base}'.")
    return pairs


def binarize(img_gray):
    """
    Binariza con Otsu; si falla o queda muy vacía/llena, fallback a adaptativa.
    """
    # Otsu
    _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)

    # Heurística para validar (evitar binarios todo blancos/negros)
    ratio_fg = np.mean(otsu == 255)
    if ratio_fg < 0.02 or ratio_fg > 0.98:
        # Adaptive (mejor para iluminación dura)
        adap = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )
        return adap
    return cv2.bitwise_and(adaptive, otsu)


def clean_binary(bin_img):
    """
    Limpieza básica: apertura y cierre suaves para quitar ruido y rellenar gaps.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed


def largest_contour(bin_img, min_area=50):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    return cnt


def crop_to_contour(bin_img, cnt, pad=2):
    x, y, w, h = cv2.boundingRect(cnt)
    y0 = max(0, y - pad); y1 = min(bin_img.shape[0], y + h + pad)
    x0 = max(0, x - pad); x1 = min(bin_img.shape[1], x + w + pad)
    roi = bin_img[y0:y1, x0:x1]
    roi = cv2.resize(roi, (NORM_W, NORM_H), interpolation=cv2.INTER_NEAREST)
    return roi


def count_holes(bin_img_roi):
    """
    Cuenta agujeros usando jerarquía de contornos (RETR_CCOMP).
    Un “agujero” es un contorno hijo (is_hole=True).
    """
    # Asegurar binario con foreground=1
    fg = (bin_img_roi > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    # hierarchy: [Next, Previous, First_Child, Parent]
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent != -1:
            holes += 1
    return holes


def symmetry_scores(roi):
    """
    Simetría horizontal y vertical como correlación con la imagen volteada.
    """
    roi_f = roi.astype(np.float32) / 255.0
    # Vertical (flip left-right)
    vflip = np.fliplr(roi_f)
    # Horizontal (flip up-down)
    hflip = np.flipud(roi_f)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        return float(np.sum(a * b) / den)

    return corr(roi_f, vflip), corr(roi_f, hflip)  # (sym_v, sym_h)


def projection_profile_features(roi):
    """
    Proyecciones horizontal y vertical: medias, desviaciones y nº de picos de “huecos”.
    Útil para distinguir caracteres con barras horizontales/verticales.
    """
    # foreground = 1
    fg = (roi > 0).astype(np.uint8)
    # perfiles como suma por filas/columnas
    vert = fg.sum(axis=0).astype(np.float32)  # ancho
    horiz = fg.sum(axis=1).astype(np.float32) # alto
    # normalizar por tamaño
    vert /= (fg.shape[0] + 1e-9)
    horiz /= (fg.shape[1] + 1e-9)

    # Para contar “valles” de tinta usamos el perfil invertido
    inv_vert = 1.0 - (vert / (vert.max() + 1e-9))
    inv_horz = 1.0 - (horiz / (horiz.max() + 1e-9))

    peaks_v, _ = find_peaks(inv_vert, height=np.mean(inv_vert) + 0.1*np.std(inv_vert))
    peaks_h, _ = find_peaks(inv_horz, height=np.mean(inv_horz) + 0.1*np.std(inv_horz))

    feats = [
        float(np.mean(vert)), float(np.std(vert)),
        float(np.mean(horiz)), float(np.std(horiz)),
        int(len(peaks_v)), int(len(peaks_h))
    ]
    return feats


def hog_simple(roi, bins=HOG_BINS):
    """
    HOG global simple (sin celdas/bloques) sobre el patch normalizado.
    """
    roi_f = roi.astype(np.float32) / 255.0
    gx = cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0  # 0..180

    hist = np.zeros(bins, dtype=np.float32)
    bin_size = 180.0 / bins
    # acumular magnitudes por ángulo
    bin_idx = np.clip((ang / bin_size).astype(np.int32), 0, bins-1)
    for b in range(bins):
        hist[b] = float(np.sum(mag[bin_idx == b]))
    # normalizar
    s = np.linalg.norm(hist) + 1e-9
    return (hist / s).astype(np.float32)


def contour_resample(cnt, n_points=256):
    pts = cnt[:, 0, :].astype(np.float32)
    # cerrar contorno si está abierto
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    s = np.hstack([0, np.cumsum(d)])
    s_uniform = np.linspace(0, s[-1], n_points)
    x_res = np.interp(s_uniform, s, pts[:, 0])
    y_res = np.interp(s_uniform, s, pts[:, 1])
    return np.vstack([x_res, y_res]).T


def curvature_features(pts):
    x, y = pts[:, 0], pts[:, 1]
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    k = np.abs(dx * ddy - dy * ddx) / denom
    return float(np.mean(k)), float(np.std(k)), int(len(find_peaks(k, height=np.mean(k)*1.5)[0]))


def radial_signature_features(pts):
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    r = np.sqrt((pts[:, 0]-cx)**2 + (pts[:, 1]-cy)**2)
    r_norm = r / (np.max(r) + 1e-9)
    return float(np.mean(r_norm)), float(np.std(r_norm))


def fourier_descriptors(pts, K=FOURIER_K):
    z = pts[:, 0] + 1j*pts[:, 1]
    Z = np.fft.fft(z)
    Z = np.abs(Z[1:K+1])  # saltar DC
    Z_norm = Z / (np.max(Z) + 1e-9)
    return Z_norm.astype(np.float32)


def shape_measures(cnt, roi):
    """
    Medidas geométricas básicas: circularidad, aspect ratio, extent, rectangularidad, solidez.
    """
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True) + 1e-9
    circularity = float(area / (perim**2))  # invariante a escala

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = float(w / (h + 1e-9))
    extent = float(area / (w*h + 1e-9))

    # Rectangularidad: área / área del mínimo rectángulo rotado
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    wmr = np.linalg.norm(box[0]-box[1])
    hmr = np.linalg.norm(box[1]-box[2])
    rect_area = float(max(1.0, wmr*hmr))
    rectangularity = float(area / rect_area)

    # Solidez: área / área del casco convexo
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull) + 1e-9)
    solidity = float(area / hull_area)

    # Espesor medio y std usando distance transform en ROI
    roi_fg = (roi > 0).astype(np.uint8)
    dist = cv2.distanceTransform(roi_fg, distanceType=cv2.DIST_L2, maskSize=3)
    # muestreamos solo donde hay foreground
    dvals = dist[roi_fg > 0]
    thickness_mean = float(np.mean(dvals) * 2.0)  # ~ diámetro local promedio
    thickness_std  = float(np.std(dvals) * 2.0)

    return [circularity, aspect, extent, rectangularity, solidity, thickness_mean, thickness_std]


# =============================================================
# EXTRACCIÓN DE FEATURES
# =============================================================

def extract_features_from_image(tipo, img_path, cls_label):
    """
    Lee, binariza, limpia, encuentra ROI y extrae todas las características.
    Retorna (label, feature_vector) o None si falla.
    """
    if tipo == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   ⚠️ No se pudo leer: {img_path}")
            return None
    else:
        img = img_path
    # Binarización y limpieza
    bin0 = binarize(img)
    bin1 = clean_binary(bin0)

    # Contorno principal
    cnt = largest_contour(bin1)
    if cnt is None:
        # puede que esté invertido (tinta blanca sobre negro). Intentar invertir.
        bin_inv = cv2.bitwise_not(bin1)
        cnt = largest_contour(bin_inv)
        if cnt is None:
            return None
        bin1 = bin_inv

    # ROI normalizado
    roi = crop_to_contour(bin1, cnt, pad=2)

    # ======= Features =======
    # 1) Medidas geométricas
    geom = shape_measures(cnt, roi)

    # 2) Huecos internos
    n_holes = count_holes(roi)

    # 3) Simetrías
    sym_v, sym_h = symmetry_scores(roi)

    # 4) Perfiles de proyección
    prof_feats = projection_profile_features(roi)

    # 5) Rasgos de contorno invariante (centrado/normalizado)
    pts = contour_resample(cnt, n_points=256)
    pts = pts - np.mean(pts, axis=0)
    pts = pts / (np.linalg.norm(pts) + 1e-9)

    mean_k, std_k, n_peaks = curvature_features(pts)
    mean_r, std_r = radial_signature_features(pts)
    fd = fourier_descriptors(pts, K=FOURIER_K)

    # 6) Hu moments (log escale para estabilidad)
    Hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    Hu = np.sign(Hu) * np.log10(np.abs(Hu) + 1e-12)

    # 7) HOG global sobre ROI normalizado
    hog = hog_simple(roi, bins=HOG_BINS)

    # Vector final (orden estable y documentado)
    features = np.hstack([
        # Etiquetas geométricas
        geom,                 # [circularity, aspect, extent, rectangularity, solidity, thick_mean, thick_std]
        n_holes,              # entero (se castea abajo)
        sym_v, sym_h,         # simetrías
        prof_feats,           # perfiles (6 valores)
        mean_k, std_k, n_peaks,
        mean_r, std_r,
        fd,                   # Fourier K
        Hu,                   # Hu 7
        hog                   # HOG bins
    ]).astype(np.float32)

    return cls_label, features


def build_excel_header(worksheet):
    """
    Escribe encabezados legibles en el Excel para trazabilidad.
    """
    headers = ([
        "label",
        "circularity","aspect_ratio","extent","rectangularity","solidity","thick_mean","thick_std",
        "n_holes",
        "sym_vert","sym_horiz",
        "proj_vert_mean","proj_vert_std","proj_horiz_mean","proj_horiz_std","proj_valleys_vert","proj_valleys_horiz",
        "curv_mean","curv_std","curv_n_peaks",
        "rad_mean","rad_std",
    ] +
    [f"FD_{i+1}" for i in range(FOURIER_K)] +
    [f"Hu_{i+1}" for i in range(7)] +
    [f"HOG_{i+1}" for i in range(HOG_BINS)]
    )

    for j, h in enumerate(headers):
        worksheet.write(0, j, h)


def extract_all_to_excel():
    pairs = list_augmented_images(OUTPUTS_BASE)
    if not pairs:
        print("⚠️ No hay imágenes aumentadas para procesar.")
        return

    workbook = xlsxwriter.Workbook(EXCEL_NAME)
    ws = workbook.add_worksheet("caracts")

    build_excel_header(ws)
    row = 1

    processed = 0
    for cls_label, img_path in pairs:
        res = extract_features_from_image(img_path, cls_label)
        if res is None:
            continue
        label, features = res

        # Escribir fila
        ws.write(row, 0, label)
        for j, val in enumerate(features, start=1):
            ws.write(row, j, float(val))
        row += 1
        processed += 1

        if processed % 200 == 0:
            print(f"   …{processed} imágenes procesadas")

    workbook.close()
    print(f"✅ Extracción terminada. Total procesadas: {processed}")
    print(f"📄 Archivo Excel: {EXCEL_NAME}")

if __name__ == "__main__":
    extract_all_to_excel()
