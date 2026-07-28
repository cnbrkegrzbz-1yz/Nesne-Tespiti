"""
Sonuç Görselleştirme
====================
Bulunan nesneleri görsel üzerinde işaretleyerek annotated görsel oluşturur.
"""

import cv2
import numpy as np

def draw_results(image, results, show_scores=True):
    """
    Bulunan nesneleri görsel üzerinde çizer.
    
    Her nesne için:
    - Yeşil bounding box
    - Sıra numarası
    - Opsiyonel: DINOv2 similarity skoru
    
    Args:
        image: BGR formatlı numpy array (orijinal yığın görseli)
        results: list of dict, her biri:
            - 'mask': SAM 2 mask dict ('bbox', 'segmentation')
            - 'similarity': float
        show_scores: True ise similarity skorlarını da göster
        
    Returns:
        numpy array: Annotated BGR görsel
    """
    annotated = image.copy()
    
    # Renk paleti — farklı nesneleri ayırt etmek için
    colors = [
        (0, 255, 0),    # Yeşil
        (0, 255, 255),  # Sarı
        (255, 200, 0),  # Açık mavi
        (0, 200, 255),  # Turuncu
        (255, 0, 255),  # Magenta
    ]

    for idx, result in enumerate(results):
        mask_dict = result['mask']
        similarity = result.get('similarity', 0.0)
        
        bbox = mask_dict['bbox']  # [x, y, w, h]
        x, y, w, h = [int(v) for v in bbox]
        
        color = colors[idx % len(colors)]
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Yarı-saydam maske overlay (opsiyonel, nesneyi vurgular)
        if 'segmentation' in mask_dict:
            seg = mask_dict['segmentation']
            if seg.shape[:2] == annotated.shape[:2]:
                overlay = annotated.copy()
                overlay[seg] = [
                    min(255, c + 60) for c in color
                ]
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        label_num = f"{idx + 1}"
        if show_scores:
            label = f"{label_num} ({similarity:.2f})"
        else:
            label = label_num
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        label_y = max(text_h + 10, y - 8)
        label_x = x
        
        cv2.rectangle(
            annotated,
            (label_x, label_y - text_h - 6),
            (label_x + text_w + 6, label_y + 4),
            (0, 0, 0), -1
        )
        
        cv2.putText(
            annotated, label,
            (label_x + 3, label_y - 2),
            font, font_scale, (255, 255, 255), thickness
        )

    # Toplam sayıyı sağ alt köşeye yaz
    total_label = f"Toplam: {len(results)} adet"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3
    (tw, th), _ = cv2.getTextSize(total_label, font, font_scale, thickness)
    
    img_h, img_w = annotated.shape[:2]
    tx = img_w - tw - 20
    ty = img_h - 20
    
    cv2.rectangle(annotated, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
    cv2.putText(annotated, total_label, (tx, ty), font, font_scale, (0, 255, 0), thickness)

    return annotated
