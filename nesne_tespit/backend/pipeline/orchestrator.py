"""
Pipeline Orkestratörü — Ana Yönetici
=====================================
5 aşamalı "Segment → Match → Count" pipeline'ını tek bir sınıfta birleştirir.

Akış:
  N Referans → [Aşama 1: DNA Çıkar (her açı için)] ────┐
                                                         │
  Yığın (tek) → [Aşama 2: SAM 2 Segmente Et]           │
          → [Aşama 3: Geometrik Filtre] ←── ref profiller ┤
          → [Aşama 4: DINOv2 Eşleştir] ←── ref embed'ler ┘
          → [Aşama 5: NMS + Çiz]
          → Sonuç: (adet, annotated_görsel)

Not: Referans tarafı çoklu-açı (N görsel) destekler, yığın tarafı bilinçli
olarak tek görsel — aynı yığının farklı açılardan fotoğrafları basitçe
toplanamaz (aynı nesne 2 fotoğrafta görünürse çift sayılır).
"""

import threading
import time

import torch

from pipeline.reference_processor import ReferenceProcessor
from pipeline.segmentor import Segmentor
from pipeline.geometric_filter import GeometricFilter
from pipeline.visual_matcher import VisualMatcher
from pipeline.post_processor import PostProcessor

class SayimPipeline:
    """
    Tüm sayım pipeline'ını yöneten ana sınıf.
    
    Modelleri bir kez yükler, her sayım isteğinde pipeline'ı çalıştırır.
    if/else router yok — tek bir pipeline hem vida hem çubuk senaryosunu kapsar.
    """

    def __init__(self, sam2_checkpoint, sam2_config, device="cuda:0"):
        """
        Modelleri yükler ve pipeline bileşenlerini oluşturur.
        
        Args:
            sam2_checkpoint: SAM 2 model ağırlıkları dosya yolu
            sam2_config: SAM 2 konfigürasyon dosyası yolu
            device: PyTorch cihazı
        """
        print("=" * 60)
        print("AKILLI SAYIM SİSTEMİ — SMC Pipeline v2")
        print("Segment → Match → Count")
        print("=" * 60)

        load_start = time.time()

        # Model yükleme
        from models.sam2_loader import SAM2Model
        from models.dinov2_loader import DINOv2Model

        self.dinov2 = DINOv2Model(device=device)
        self.sam2 = SAM2Model(
            checkpoint_path=sam2_checkpoint,
            config_path=sam2_config,
            device=device
        )

        # Pipeline bileşenlerini oluştur
        self.ref_processor = ReferenceProcessor(self.dinov2)
        self.segmentor = Segmentor(self.sam2)
        self.geo_filter = GeometricFilter(
            aspect_ratio_tolerance=0.65,
            solidity_tolerance=0.40,
            min_area=30,
            max_area_ratio=20.0,
        )
        # NOT: fallback_threshold önceden 0.55'ti — dört ayrı gerçek testte de
        # (temiz/kirli referans, hizalı/dağınık yığın fark etmeksizin) gerçek
        # eşleşmelerin skoru hep 0.34-0.55 bandında kaldı ve "belirgin boşluk
        # yok" durumunda bu ulaşılamaz sabite düşülünce sürekli 0 sonuç
        # çıktı. min_threshold (0.35) ile aynı yapıldı — algoritmanın zaten
        # var olan güvenlik tabanından daha yüksek, veriyle kalibre
        # edilmemiş bir bar icat etmesin diye.
        self.matcher = VisualMatcher(
            self.dinov2,
            min_threshold=0.35,
            fallback_threshold=0.35,
        )
        self.post_processor = PostProcessor(
            nms_iou_threshold=0.3,
            show_scores=True,
        )

        # SAM2'nin otomatik maske üreticisi (SAM2ImagePredictor) stateful ve
        # thread-safe DEĞİL — set_image()+predict() adımları paylaşılan tek
        # bir nesne üzerinde çalışıyor. main.py'deki endpoint'ler `def`
        # (async değil) olduğu için FastAPI birden fazla isteği GERÇEKTEN
        # paralel thread'lerde çalıştırabiliyor; bu kilit olmadan iki eşzamanlı
        # sayım isteği SAM2'nin iç durumunu birbirine karıştırıp rastgele
        # hatalara (ör. "image must be set", şekil uyuşmazlığı) yol açar.
        self._kilit = threading.Lock()

        load_time = time.time() - load_start
        print(f"\n[SİSTEM] Tüm modeller {load_time:.1f} saniyede yüklendi.")
        print("[SİSTEM] Pipeline hazır, sayım bekleniyor...")
        print("=" * 60)

    def sayim_yap(self, referanslar, yigin, metin=""):
        """
        Ana sayım fonksiyonu — 5 aşamalı pipeline'ı çalıştırır.

        Args:
            referanslar: BGR formatlı numpy array LİSTESİ (N adet referans
                fotoğrafı, farklı açılardan "öğretim" için)
            yigin: BGR formatlı numpy array (yığın fotoğrafı, tek görsel)
            metin: Kullanıcı metin prompt'u (opsiyonel, sadece loglama için)

        Returns:
            tuple: (adet: int, annotated_görsel: numpy array)
        """
        # Aynı anda tek bir sayım çalışsın diye kilit — bkz. __init__'teki not.
        # Kilit beklerken başka bir istek burada bekler, hata almaz.
        with self._kilit:
            pipeline_start = time.time()

            if metin:
                print(f"\n{'─' * 40}")
                print(f"YENİ SAYIM İSTEĞİ: '{metin}'")
                print(f"{'─' * 40}")
            else:
                print(f"\n{'─' * 40}")
                print(f"YENİ SAYIM İSTEĞİ (metin prompt yok)")
                print(f"{'─' * 40}")

            # AŞAMA 1: N Referans DNA Çıkarımı
            ref_profiles = self.ref_processor.process_batch(referanslar)
            ref_geometries = [p['geometry'] for p in ref_profiles]
            ref_embeddings = torch.cat([p['embedding'] for p in ref_profiles], dim=0)  # (K, 768)

            # AŞAMA 2: Yığın Segmentasyonu
            all_masks = self.segmentor.segment(yigin, multi_scale=True)

            # AŞAMA 3: Geometrik Ön-Filtreleme
            candidates = self.geo_filter.filter(all_masks, ref_geometries)

            # AŞAMA 4: DINOv2 Görsel Eşleştirme
            matches = self.matcher.match(yigin, candidates, ref_embeddings)

            # AŞAMA 5: NMS + Çizim
            final_results, annotated_image = self.post_processor.finalize(yigin, matches)

            # Sonuç Raporu
            pipeline_time = time.time() - pipeline_start
            adet = len(final_results)

            print(f"\n{'=' * 40}")
            print(f"SONUÇ: {adet} adet nesne bulundu")
            print(f"Pipeline süresi: {pipeline_time:.2f} saniye")
            print(f"  Kullanılan referans profili: {len(ref_profiles)}")
            print(f"  Aşama detay: {len(all_masks)} maske → "
                  f"{len(candidates)} aday → "
                  f"{len(matches)} eşleşme → "
                  f"{adet} sonuç")
            print(f"{'=' * 40}\n")

            return adet, annotated_image
