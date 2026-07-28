"""
AŞAMA 3 — Geometrik Ön-Filtreleme
===================================
SAM 2'nin ürettiği tüm maskeleri, referans nesnenin geometrik profiline
göre hızlıca eler. DINOv2'ye gidecek aday sayısını %60-80 azaltarak
hem hızı artırır hem de false positive'leri düşürür.

Filtre kriterleri:
- Aspect ratio benzerliği (referans ± tolerans)
- Minimum/maksimum alan eşiği
- Solidity benzerliği (şekil tutarlılığı)
"""

import numpy as np

from utils.mask_utils import compute_geometric_properties


class GeometricFilter:
    """
    Geometrik özelliklere göre aday maskeleri filtreler.
    
    Bu bir 'sert' karar değil, 'ön-eleme'dir. Toleranslar bilinçli
    olarak geniş tutulur. Asıl karar Aşama 4'te DINOv2 ile verilir.
    """

    def __init__(
        self,
        aspect_ratio_tolerance=0.65,
        solidity_tolerance=0.40,
        min_area=30,
        max_area_ratio=20.0,
    ):
        """
        Args:
            aspect_ratio_tolerance: AR farkı bu oranın altındaysa geçer (0.65 = %65)
            solidity_tolerance: Solidity farkı bu oranın altındaysa geçer
            min_area: Bu piksel alanının altındaki maskeler atılır
            max_area_ratio: Referansın alanının bu katından büyük maskeler atılır
        """
        self.ar_tolerance = aspect_ratio_tolerance
        self.sol_tolerance = solidity_tolerance
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def filter(self, masks, ref_geometry):
        """
        Maskeleri referans geometrisine göre filtreler.
        
        Args:
            masks: SAM 2 maske listesi (Aşama 2 çıktısı)
            ref_geometry: Referans geometrik profil (Aşama 1 çıktısı)
                - aspect_ratio, solidity, area
                
        Returns:
            list[dict]: Filtrelenmiş maske listesi
        """
        ref_ar = ref_geometry.get('aspect_ratio', 1.0)
        ref_sol = ref_geometry.get('solidity', 0.85)
        ref_area = ref_geometry.get('area', 1000)

        print(f"[AŞAMA 3] Geometrik filtreleme başlatılıyor...")
        print(f"  → Referans profil: AR={ref_ar:.2f}, Sol={ref_sol:.2f}, Alan={ref_area}")
        print(f"  → Gelen maske sayısı: {len(masks)}")

        filtered = []
        stats = {'ar_rejected': 0, 'sol_rejected': 0, 'area_rejected': 0, 'no_props': 0}

        for mask in masks:
            # Geometrik özellikleri hesapla
            props = compute_geometric_properties(mask)

            if props is None:
                stats['no_props'] += 1
                continue

            # ─── Filtre 1: Minimum alan ───
            if props['area'] < self.min_area:
                stats['area_rejected'] += 1
                continue

            # ─── Filtre 2: Maksimum alan (referansın X katından büyük olamaz) ───
            if ref_area > 0 and props['area'] > (ref_area * self.max_area_ratio):
                stats['area_rejected'] += 1
                continue

            # ─── Filtre 3: Aspect ratio benzerliği ───
            if ref_ar > 0:
                ar_diff = abs(props['aspect_ratio'] - ref_ar) / ref_ar
                if ar_diff > self.ar_tolerance:
                    stats['ar_rejected'] += 1
                    continue

            # ─── Filtre 4: Solidity benzerliği ───
            if ref_sol > 0:
                sol_diff = abs(props['solidity'] - ref_sol) / ref_sol
                if sol_diff > self.sol_tolerance:
                    stats['sol_rejected'] += 1
                    continue

            # Tüm filtrelerden geçti
            mask['_geometric_props'] = props
            filtered.append(mask)

        print(f"  → Filtreleme sonucu: {len(filtered)} aday kaldı "
              f"(AR elendi: {stats['ar_rejected']}, "
              f"Sol elendi: {stats['sol_rejected']}, "
              f"Alan elendi: {stats['area_rejected']}, "
              f"Geçersiz: {stats['no_props']})")

        return filtered
