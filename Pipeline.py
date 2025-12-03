import cv2
import numpy as np
import os
import argparse
import re

# ---------------- Helper functions ----------------

def denoise_img(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def get_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower = int(max(0, 0.7*v))
    upper = int(min(255, 1.3*v))
    edges = cv2.Canny(gray, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges

def get_binary_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)
    mask = cv2.bitwise_or(otsu, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

# ---------------- Grid segmentation ----------------

def segment_grid(img, rows, cols, out_dir, base_name):
    h, w = img.shape[:2]
    piece_h, piece_w = h // rows, w // cols
    pieces = []

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for i in range(rows):
        for j in range(cols):
            y0, y1 = i*piece_h, (i+1)*piece_h
            x0, x1 = j*piece_w, (j+1)*piece_w
            piece = img[y0:y1, x0:x1]
            pieces.append(piece)
            save_image(os.path.join(out_dir, f"{base_name}_piece_{count+1}.png"), piece)
            count += 1
    return pieces

# ---------------- Main pipeline ----------------

def process_image(path, out_root, debug=False):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        print("Failed to read:", path)
        return

    folder_name = os.path.basename(os.path.dirname(path))
    match = re.search(r'(\d+)x(\d+)', folder_name)
    if match:
        rows, cols = map(int, match.groups())
    else:
        rows, cols = 1, 1

    dirs = {}
    for k in ['denoise','clahe','edges','masks','overlays','segmented']:
        dirs[k] = os.path.join(out_root, folder_name, k)
        os.makedirs(dirs[k], exist_ok=True)

    # 1. Denoise
    den = denoise_img(img)
    save_image(os.path.join(dirs['denoise'], f"{name}_denoise.png"), den)

    # 2. CLAHE
    clahe = apply_clahe(den)
    save_image(os.path.join(dirs['clahe'], f"{name}_clahe.png"), clahe)

    # 3. Edges
    edges = get_edges(clahe)
    save_image(os.path.join(dirs['edges'], f"{name}_edges.png"), edges)

    # 4. Mask
    mask = get_binary_mask(clahe)
    save_image(os.path.join(dirs['masks'], f"{name}_mask.png"), mask)

    # 5. Overlay
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
    save_image(os.path.join(dirs['overlays'], f"{name}_overlay.png"), overlay)

    # 6. Segment pieces (grid-based)
    pieces = segment_grid(img, rows, cols, dirs['segmented'], name)
    if debug:
        print(f"[{folder_name}/{name}] Processed: {len(pieces)} pieces ({rows}x{cols})")

# ---------------- Batch processing ----------------

def process_folder(input_folder, out_root, debug=False):
    for root, _, files in os.walk(input_folder):
        for fname in files:
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                path = os.path.join(root, fname)
                process_image(path, out_root, debug=debug)

# ---------------- Entry point ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jigsaw Puzzle Classical CV Pipeline")
    parser.add_argument('--input', required=True, help="Input folder with puzzles")
    parser.add_argument('--out', required=True, help="Output folder")
    parser.add_argument('--debug', action='store_true', help="Show debug prints")
    args = parser.parse_args()

    process_folder(args.input, args.out, debug=args.debug)
