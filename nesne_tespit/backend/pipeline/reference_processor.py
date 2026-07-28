"""
AŞAMA 1 — Referans Özellik Çıkarımı (Reference DNA Extraction)
================================================================
Kullanıcının yüklediği referans fotoğraftan nesnenin:
1. Görsel embedding'ini (DINOv2 — 768 boyutlu DNA vektörü)
2. Geometrik profilini (aspect ratio, solidity, circularity, alan)

çıkarır. Bu bilgiler sonraki aşamalarda eşleştirme ve filtreleme için kullanılır.
"""

import cv2
import numpy as np

from utils.image_utils import remove_background_simple
from utils.mask_utils import compute_geometric_properties


class ReferenceProcessor:
    """
    Referans fotoğraftan nesnenin görsel DNA'sını ve geometrik profilini çıkarır.
    """

    def __init__(self, dinov2_model):
        """
        Args:
            dinov2_model: Yüklenmiş DINOv2Model instance
        """
        self.dinov2 = dinov2_model

    def process(self, ref_image):
        """
        Referans fotoğrafı işleyerek nesne profilini çıkarır.
        
        İşlem adımları:
        1. Arka plan kaldır / nesneyi izole et
        2. DINOv2 embedding çıkar
        3. Geometrik profil hesapla
        
        Args:
            ref_image: BGR formatlı numpy array (referans fotoğrafı)
            
        Returns:
            dict: Referans profili
                - 'embedding': torch.Tensor (1, 768) — DINOv2 görsel DNA
                - 'geometry': dict — Geometrik özellikler
                - 'crop': numpy array — İzole edilmiş nesne görseli
        """
        print("[AŞAMA 1] Referans özellik çıkarımı başlatılıyor...")

        # Adım 1: Nesneyi izole et (arka plan kaldır)
        ref_crop, ref_mask = remove_background_simple(ref_image)
        print(f"  → Nesne izole edildi: {ref_crop.shape[1]}x{ref_crop.shape[0]} piksel")

        # Adım 2: DINOv2 embedding çıkar
        ref_embedding = self.dinov2.get_embedding(ref_crop)
        if ref_embedding is None:
            # Fallback: Orijinal görselden embedding al
            print("  → [UYARI] İzole görsel başarısız, orijinal kullanılıyor.")
            ref_embedding = self.dinov2.get_embedding(ref_image)

        # Adım 3: Geometrik profil hesapla
        geometry = self._extract_geometry(ref_crop, ref_mask)
        print(f"  → Geometrik profil: AR={geometry['aspect_ratio']:.2f}, "
              f"Solidity={geometry['solidity']:.2f}, "
              f"Alan={geometry['area']} px")

        print("[AŞAMA 1] Referans profili hazır.")

        return {
            'embedding': ref_embedding,
            'geometry': geometry,
            'crop': ref_crop,
        }

    def _extract_geometry(self, crop, mask):
        """
        Kırpılmış referans görselinden geometrik özellikler çıkarır.
        
        Args:
            crop: Kırpılmış BGR görsel
            mask: Boolean maske
            
        Returns:
            dict: Geometrik özellikler
        """
        # Maske varsa onu kullan, yoksa kontur tabanlı analiz yap
        if mask is not None and mask.any():
            # Maskeyi SAM formatına dönüştür
            mock_mask = {
                'segmentation': mask,
                'bbox': list(cv2.boundingRect(mask.astype(np.uint8) * 255)),
                'area': int(np.sum(mask)),
            }
            props = compute_geometric_properties(mock_mask)
            if props is not None:
                return props

        # Fallback: Kırpımın kendisinden hesapla
        h, w = crop.shape[:2]
        short_side = min(w, h) if min(w, h) > 0 else 1
        long_side = max(w, h)

        return {
            'aspect_ratio': long_side / short_side,
            'solidity': 0.85,  # Varsayılan (bilinmiyorsa orta değer)
            'circularity': 0.5,
            'area': w * h,
            'bbox': (0, 0, w, h),
        }
