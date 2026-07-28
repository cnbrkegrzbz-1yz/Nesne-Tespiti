"""
AŞAMA 2 — Sınıf-Bağımsız Segmentasyon Sarmalayıcısı
=====================================================
SAM 2 modelini çağırarak yığın fotoğrafındaki tüm potansiyel
nesneleri metin/prompt gerektirmeden segmente eder.

Bu modül, SAM2Model'i doğrudan pipeline'a entegre eder ve
çoklu çözünürlük tarama stratejisini yönetir.
"""


class Segmentor:
    """
    SAM 2 segmentasyonunu pipeline'a entegre eden sarmalayıcı.
    """

    def __init__(self, sam2_model):
        """
        Args:
            sam2_model: Yüklenmiş SAM2Model instance
        """
        self.sam2 = sam2_model

    def segment(self, pile_image, multi_scale=True):
        """
        Yığın fotoğrafındaki tüm potansiyel nesneleri segmente eder.
        
        Args:
            pile_image: BGR formatlı numpy array (yığın fotoğrafı)
            multi_scale: True ise çoklu çözünürlük kullan
            
        Returns:
            list[dict]: SAM 2 maske listesi (deduplike edilmiş)
        """
        print(f"[AŞAMA 2] Yığın segmentasyonu başlatılıyor... (multi_scale={multi_scale})")

        masks = self.sam2.generate_masks(pile_image, multi_scale=multi_scale)

        print(f"[AŞAMA 2] Toplam {len(masks)} potansiyel nesne segmente edildi.")
        return masks
