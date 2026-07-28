"""
AŞAMA 4 — Görsel Eşleştirme (Visual Matching)
===============================================
Geometrik filtreden geçen her aday maskeyi DINOv2 embedding'i ile
referans nesneye karşı görsel olarak karşılaştırır.

Bu aşama, HSV histogram'ın yapamadığını yapar:
- Gölgede kalan nesneleri DOĞRU eşleştirir (ışığa dirençli)
- Farklı nesne türlerini ayırt eder (semantik anlam)

Dinamik Eşik Stratejisi:
- Sabit eşik yerine, skor dağılımındaki doğal "boşluğu" (gap) bulur
- Farklı aydınlatma koşullarına otomatik adapte olur
"""

import torch
import numpy as np

from utils.image_utils import crop_with_mask


class VisualMatcher:
    """
    DINOv2 tabanlı görsel eşleştirici.
    Aday nesneleri referansla karşılaştırarak eşleşenleri belirler.
    """

    def __init__(self, dinov2_model, min_threshold=0.35, fallback_threshold=0.55):
        """
        Args:
            dinov2_model: Yüklenmiş DINOv2Model instance
            min_threshold: Dinamik eşik hiçbir zaman bunun altına düşemez
            fallback_threshold: Yeterli veri yoksa kullanılacak sabit eşik
        """
        self.dinov2 = dinov2_model
        self.min_threshold = min_threshold
        self.fallback_threshold = fallback_threshold

    def match(self, pile_image, candidate_masks, ref_embedding):
        """
        Aday maskeleri referans embedding ile karşılaştırır.
        
        İşlem:
        1. Her aday maskeyi pile_image'dan kırp
        2. Batch olarak DINOv2 embedding çıkar
        3. Cosine similarity hesapla
        4. Dinamik eşik belirle
        5. Eşiğin üzerindeki adayları döndür
        
        Args:
            pile_image: BGR formatlı numpy array (yığın fotoğrafı)
            candidate_masks: Filtrelenmiş maske listesi (Aşama 3 çıktısı)
            ref_embedding: Referans DINOv2 embedding (1, 768) — Aşama 1 çıktısı
            
        Returns:
            list[dict]: Eşleşen sonuçlar, her biri:
                - 'mask': SAM 2 mask dict
                - 'similarity': float (cosine similarity skoru)
        """
        print(f"[AŞAMA 4] Görsel eşleştirme başlatılıyor... ({len(candidate_masks)} aday)")

        if not candidate_masks or ref_embedding is None:
            print("[AŞAMA 4] Aday veya referans yok, boş sonuç döndürülüyor.")
            return []

        # Adım 1: Tüm adayları kırp
        crops = []
        valid_mask_indices = []
        for idx, mask in enumerate(candidate_masks):
            crop = crop_with_mask(pile_image, mask, padding=5)
            if crop is not None and crop.size > 0:
                crops.append(crop)
                valid_mask_indices.append(idx)

        if not crops:
            print("[AŞAMA 4] Geçerli kırpım yok, boş sonuç döndürülüyor.")
            return []

        print(f"  → {len(crops)} geçerli kırpım hazırlandı, batch embedding başlıyor...")

        # Adım 2: Batch olarak DINOv2 embedding çıkar
        candidate_embeddings, embed_valid_indices = self.dinov2.get_embeddings_batch(crops)

        if candidate_embeddings.shape[0] == 0:
            print("[AŞAMA 4] Embedding çıkarılamadı, boş sonuç döndürülüyor.")
            return []

        # Adım 3: Cosine similarity hesapla (matris çarpımı ile hızlı)
        # ref_embedding: (1, 768), candidate_embeddings: (N, 768)
        similarities = torch.mm(ref_embedding, candidate_embeddings.t()).squeeze(0)
        similarities = similarities.cpu().numpy()

        print(f"  → Similarity aralığı: [{similarities.min():.3f}, {similarities.max():.3f}]")

        # Adım 4: Dinamik eşik belirle
        threshold = self._determine_dynamic_threshold(similarities)
        print(f"  → Dinamik eşik: {threshold:.3f}")

        # Adım 5: Eşleşenleri topla
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                # embed_valid_indices[i] → crops listesindeki indeks
                # valid_mask_indices[crops_idx] → candidate_masks'taki indeks
                crop_idx = embed_valid_indices[i]
                mask_idx = valid_mask_indices[crop_idx]

                results.append({
                    'mask': candidate_masks[mask_idx],
                    'similarity': float(sim),
                })

        print(f"[AŞAMA 4] Eşleşme sonucu: {len(results)} nesne eşleşti "
              f"(eşik={threshold:.3f})")

        return results

    def _determine_dynamic_threshold(self, scores):
        """
        Skor dağılımındaki doğal boşluğu (gap) bularak otomatik eşik belirler.
        
        Mantık:
        - Doğru eşleşmeler genelde birbirine yakın yüksek skorlar üretir
        - Yanlış eşleşmeler düşük skorlar üretir
        - Bu iki küme arasındaki en büyük boşluk = en iyi eşik noktası
        
        Bu sayede farklı aydınlatma koşullarına otomatik adapte olur.
        HSV histogram'ın gölgede çökmesinin aksine, DINOv2 skorları
        ışık değişimine dirençli olduğu için gap daha belirgin olur.
        
        Args:
            scores: numpy array — cosine similarity skorları
            
        Returns:
            float: Dinamik eşik değeri
        """
        if len(scores) < 2:
            return self.fallback_threshold

        scores_sorted = np.sort(scores)[::-1]  # Azalan sıra

        # En büyük boşluğu (gap) bul
        max_gap = 0
        gap_idx = 0

        for i in range(len(scores_sorted) - 1):
            gap = scores_sorted[i] - scores_sorted[i + 1]
            if gap > max_gap:
                max_gap = gap
                gap_idx = i

        # Boşluk yeterince büyük mü? (en az 0.05 olmalı)
        if max_gap < 0.05:
            # Belirgin bir ayrım yok — tüm skorlar benzer
            # Bu durumda fallback eşik kullan
            return self.fallback_threshold

        # Eşik: Boşluğun ortası
        threshold = (scores_sorted[gap_idx] + scores_sorted[gap_idx + 1]) / 2

        # Minimum güvenlik eşiği
        return max(threshold, self.min_threshold)
