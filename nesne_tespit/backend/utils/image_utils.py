"""
Görsel İşleme Yardımcıları
==========================
Kırpma, boyutlandırma ve format dönüşüm fonksiyonları.
"""

import cv2
import numpy as np

def crop_with_mask(image, mask_dict, padding=5):
    """
    SAM 2 maskesi kullanarak nesneyi görselten kırpar.
    Maskenin dışındaki pikselleri siyah yapar (temiz kırpım).
    
    Args:
        image: BGR formatlı numpy array
        mask_dict: SAM 2 maske dict'i ('segmentation', 'bbox' içermeli)
        padding: Kırpım etrafına eklenecek piksel boşluğu
        
    Returns:
        numpy array: Kırpılmış BGR görsel veya None
    """
    if image is None or mask_dict is None:
        return None

    segmentation = mask_dict['segmentation']
    h, w = image.shape[:2]

    # Maske boyutu görsel boyutuyla uyuşmalı
    if segmentation.shape[:2] != (h, w):
        segmentation = cv2.resize(
            segmentation.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    mask_3ch = np.stack([segmentation] * 3, axis=-1)
    masked_image = np.where(mask_3ch, image, 0).astype(np.uint8)

    bbox = mask_dict['bbox']  # [x, y, w, h]
    x, y, bw, bh = [int(v) for v in bbox]

    # Padding ekle (görsel sınırlarını aşma)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    crop = masked_image[y1:y2, x1:x2]

    if crop.size == 0:
        return None
    return crop

def crop_with_bbox(image, bbox, padding=5):
    """
    Bounding box koordinatları ile kırpar (maske olmadan).
    
    Args:
        image: BGR formatlı numpy array
        bbox: [x, y, w, h] formatında bbox
        padding: Kırpım etrafına eklenecek piksel boşluğu
        
    Returns:
        numpy array: Kırpılmış BGR görsel veya None
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None
    return crop

def safe_resize(image, target_size=(224, 224)):
    """
    Görseli güvenli şekilde boyutlandırır.
    Boş görsellerde crash önler.
    
    Args:
        image: numpy array
        target_size: (width, height) tuple
        
    Returns:
        numpy array veya None
    """
    if image is None or image.size == 0:
        return None

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def remove_background_simple(image):
    """
    Basit arka plan kaldırma — referans fotoğraf için.
    Merkezdeki nesneyi otomatik kırpar.
    
    rembg kurulu değilse veya başarısız olursa,
    kontur tabanlı fallback kullanır.
    
    Args:
        image: BGR formatlı numpy array
        
    Returns:
        tuple: (kırpılmış_görsel, maske)
    """
    # Önce rembg'yi dene (varsa)
    try:
        from rembg import remove
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Arka planı kaldır (RGBA döner)
        result = remove(pil_img)
        result_np = np.array(result)

        # Alpha kanalından maske çıkar
        if result_np.shape[2] == 4:
            mask = result_np[:, :, 3] > 128
            # RGBA → BGR
            bgr_result = cv2.cvtColor(result_np[:, :, :3], cv2.COLOR_RGB2BGR)

            # Maske ile kırp
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                cropped = bgr_result[y1:y2+1, x1:x2+1]
                mask_cropped = mask[y1:y2+1, x1:x2+1]
                return cropped, mask_cropped

    except ImportError:
        print("[UYARI] rembg kurulu değil, kontur tabanlı fallback kullanılıyor.")
    except Exception as e:
        print(f"[UYARI] rembg başarısız: {e}, fallback kullanılıyor.")

    # Fallback: Kontur tabanlı merkez nesne bulma
    return _fallback_center_crop(image)

def _fallback_center_crop(image):
    """
    rembg yoksa, merkezdeki en büyük nesneyi kontur ile bulur.
    
    Args:
        image: BGR formatlı numpy array
        
    Returns:
        tuple: (kırpılmış_görsel, maske)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur ile gürültüyü azalt
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfolojik temizlik
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Hiçbir kontur bulunamazsa, görselin merkez %60'ını kırp
        h, w = image.shape[:2]
        margin_x, margin_y = int(w * 0.2), int(h * 0.2)
        crop = image[margin_y:h-margin_y, margin_x:w-margin_x]
        mask = np.ones(crop.shape[:2], dtype=bool)
        return crop, mask

    # Merkeze en yakın ve yeterince büyük konturu bul
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    best_cnt = None
    best_score = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.005):  # Çok küçükleri atla (%0.5'ten küçük)
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Mesafe skoru (merkeze yakınlık)
        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        # Büyük nesneleri tercih et (alan ile ters orantılı bonus)
        score = distance / (area ** 0.3)

        if score < best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        h, w = image.shape[:2]
        margin_x, margin_y = int(w * 0.2), int(h * 0.2)
        crop = image[margin_y:h-margin_y, margin_x:w-margin_x]
        mask = np.ones(crop.shape[:2], dtype=bool)
        return crop, mask

    # En iyi konturun bbox'ı ile kırp
    x, y, bw, bh = cv2.boundingRect(best_cnt)
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    crop = image[y1:y2, x1:x2]

    # Maske oluştur
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_full, [best_cnt], -1, 255, -1)
    mask_crop = mask_full[y1:y2, x1:x2].astype(bool)

    return crop, mask_crop
