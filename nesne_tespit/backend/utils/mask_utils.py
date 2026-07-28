"""
Maske İşleme Yardımcıları
=========================
SAM 2 maskeleri için IoU hesaplama, NMS ve format dönüşüm fonksiyonları.
"""

import numpy as np
import cv2


def mask_to_bbox(mask_segmentation):
    """
    Boolean segmentation maskesinden bounding box çıkarır.
    
    Args:
        mask_segmentation: Boolean numpy array (2D)
        
    Returns:
        tuple: (x, y, w, h) veya None
    """
    if mask_segmentation is None:
        return None

    ys, xs = np.where(mask_segmentation)
    if len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def compute_iou(mask_a, mask_b):
    """
    İki boolean maske arasındaki IoU (Intersection over Union) değerini hesaplar.
    
    Args:
        mask_a: Boolean numpy array
        mask_b: Boolean numpy array
        
    Returns:
        float: [0.0, 1.0] aralığında IoU değeri
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_iou_bbox(bbox_a, bbox_b):
    """
    İki bounding box arasındaki IoU değerini hesaplar.
    
    Args:
        bbox_a: (x, y, w, h)
        bbox_b: (x, y, w, h)
        
    Returns:
        float: IoU değeri
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Kesişim alanı
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def apply_nms(results, iou_threshold=0.3):
    """
    Non-Maximum Suppression — Örtüşen sonuçları eler.
    En yüksek similarity skorlu sonuç korunur.
    
    Args:
        results: list of dict, her biri şunları içermeli:
            - 'mask': SAM 2 mask dict
            - 'similarity': float (DINOv2 cosine similarity)
        iou_threshold: Bu eşiğin üzerinde örtüşenlerden düşük skrolu eler
        
    Returns:
        list of dict: Filtrelenmiş sonuçlar
    """
    if len(results) <= 1:
        return results

    # Similarity skoruna göre azalan sırala
    sorted_results = sorted(results, key=lambda r: r['similarity'], reverse=True)

    keep = []
    suppressed = set()

    for i, result_a in enumerate(sorted_results):
        if i in suppressed:
            continue
        keep.append(result_a)

        bbox_a = result_a['mask']['bbox']

        for j in range(i + 1, len(sorted_results)):
            if j in suppressed:
                continue

            bbox_b = sorted_results[j]['mask']['bbox']
            iou = compute_iou_bbox(bbox_a, bbox_b)

            if iou > iou_threshold:
                suppressed.add(j)

    return keep


def compute_geometric_properties(mask_dict):
    """
    Bir SAM 2 maskesinden geometrik özellikleri çıkarır.
    
    Args:
        mask_dict: SAM 2 maske dict'i
        
    Returns:
        dict: Geometrik özellikler
            - aspect_ratio: Genişlik/yükseklik oranı (her zaman >= 1)
            - solidity: Alan / Convex hull alanı (0-1, 1=tamamen konveks)
            - circularity: Dairesellik (4π×alan / çevre², 1=mükemmel daire)
            - area: Piksel alanı
            - bbox: (x, y, w, h)
    """
    segmentation = mask_dict['segmentation']

    # Kontur bul
    seg_uint8 = segmentation.astype(np.uint8) * 255
    contours, _ = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # En büyük konturu al
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 1:
        return None

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect ratio (her zaman >= 1 olacak şekilde)
    short_side = min(w, h) if min(w, h) > 0 else 1
    long_side = max(w, h)
    aspect_ratio = long_side / short_side

    # Solidity (doluluk oranı)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Circularity (dairesellik)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    return {
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'circularity': circularity,
        'area': area,
        'bbox': (x, y, w, h),
    }
