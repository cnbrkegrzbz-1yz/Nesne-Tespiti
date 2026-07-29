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

    def _gecer_mi(self, props, ref_geometry):
        """
        Tek bir aday, tek bir referans profiline karşı TÜM kriterleri
        (alan/aspect-ratio/solidity) birlikte geçiyor mu kontrol eder.

        Returns:
            bool: Bu referans profiline göre aday geçerli mi
        """
        ref_ar = ref_geometry.get('aspect_ratio', 1.0)
        ref_sol = ref_geometry.get('solidity', 0.85)
        ref_area = ref_geometry.get('area', 1000)

        # Maksimum alan (referansın X katından büyük olamaz)
        if ref_area > 0 and props['area'] > (ref_area * self.max_area_ratio):
            return False

        # Aspect ratio benzerliği
        if ref_ar > 0:
            ar_diff = abs(props['aspect_ratio'] - ref_ar) / ref_ar
            if ar_diff > self.ar_tolerance:
                return False

        # Solidity benzerliği
        if ref_sol > 0:
            sol_diff = abs(props['solidity'] - ref_sol) / ref_sol
            if sol_diff > self.sol_tolerance:
                return False

        return True

    def filter(self, masks, ref_geometries):
        """
        Maskeleri referans geometri(ler)ine göre filtreler.

        Çoklu referans mantığı: bir aday, N referans açısından EN AZ
        BİRİYLE tüm kriterleri (alan+AR+solidity) birlikte geçerse tutulur
        (referanslar arası OR, bir referansın kendi kriterleri arası AND).
        Bu şekilde farklı açılardan (üstten/yandan) çekilmiş referanslar
        çok farklı geometrik profillere sahip olsa bile hepsi ayrı ayrı
        değerlendirilir — "her filtreyi ayrı referanstan geçmek" gibi
        anlamsız kombinasyonlara izin verilmez.

        Args:
            masks: SAM 2 maske listesi (Aşama 2 çıktısı)
            ref_geometries: Referans geometrik profil listesi (Aşama 1 çıktısı)
                Her biri: aspect_ratio, solidity, area

        Returns:
            list[dict]: Filtrelenmiş maske listesi
        """
        print(f"[AŞAMA 3] Geometrik filtreleme başlatılıyor...")
        print(f"  → Referans profil sayısı: {len(ref_geometries)}")
        for i, g in enumerate(ref_geometries):
            print(f"    #{i + 1}: AR={g.get('aspect_ratio', 1.0):.2f}, "
                  f"Sol={g.get('solidity', 0.85):.2f}, Alan={g.get('area', 0)}")
        print(f"  → Gelen maske sayısı: {len(masks)}")

        filtered = []
        stats = {'ar_sol_area_rejected': 0, 'area_rejected': 0, 'no_props': 0}

        for mask in masks:
            props = compute_geometric_properties(mask)

            if props is None:
                stats['no_props'] += 1
                continue

            # Global filtre: minimum alan (hangi referansla kıyaslanırsa
            # kıyaslansın değişmez, tek seferde elenir)
            if props['area'] < self.min_area:
                stats['area_rejected'] += 1
                continue

            # En az bir referans profiliyle tüm kriterleri birlikte geçiyor mu?
            if any(self._gecer_mi(props, ref_geom) for ref_geom in ref_geometries):
                mask['_geometric_props'] = props
                filtered.append(mask)
            else:
                stats['ar_sol_area_rejected'] += 1

        print(f"  → Filtreleme sonucu: {len(filtered)} aday kaldı "
              f"(hiçbir referansla uyuşmadı: {stats['ar_sol_area_rejected']}, "
              f"min. alan altı: {stats['area_rejected']}, "
              f"geçersiz: {stats['no_props']})")

        return filtered
