"""
SAM 2 Model Yükleyici — Sınıf-Bağımsız Segmentasyon Motoru
==========================================================
Meta'nın SAM 2 (Segment Anything Model 2) modelini kullanarak
görseldeki TÜM nesneleri metin/prompt gerektirmeden segmente eder.

Neden SAM 2 (YOLO-World değil)?
- YOLO-World: "gray plastic rod" gibi metin prompt'a bağımlı → Prompt Brittleness
- SAM 2: "Burada bir nesne var" der, ne olduğunu sormaz → Prompt bağımsız

Çoklu Çözünürlük Stratejisi:
- Orijinal çözünürlük: Büyük nesneler (çubuklar, borular)
- 2x upscale: Küçük nesneler (vidalar, somunlar — 20-30px → 40-60px)
- IoU tabanlı deduplication: Çift sayımı önler
"""

import torch
import numpy as np
import cv2


class SAM2Model:
    """
    SAM 2 Automatic Mask Generator sarmalayıcısı.
    
    Metin/etiket gerektirmeden görseldeki tüm potansiyel nesneleri
    piksel-düzeyinde maskeler. Çoklu çözünürlük desteği ile
    hem milimetrik vidaları hem uzun çubukları yakalar.
    """

    def __init__(self, checkpoint_path, config_path, device="cuda:0"):
        """
        Args:
            checkpoint_path: SAM 2 model ağırlıkları (.pt dosyası)
                             Örn: "checkpoints/sam2.1_hiera_large.pt"
            config_path: SAM 2 model konfigürasyonu
                         Örn: "configs/sam2.1/sam2.1_hiera_l.yaml"
            device: PyTorch cihazı ("cuda:0" veya "cpu")
        """
        print(f"[MODÜL] SAM 2 başlatılıyor... Hedef: {device}")
        self.device = torch.device(device)

        # TF32 optimizasyonu — Ada Lovelace (SM 8.9) ve üzeri için
        # NOT: Bu global ayarlar autocast'ten farklı — sadece matris çarpımı
        # hassasiyetini etkiler, güvenlidir
        self._configure_gpu_optimizations()

        # SAM 2 modelini oluştur
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_model = build_sam2(
            config_path,
            checkpoint_path,
            device=str(self.device),
            apply_postprocessing=True  # Maske kenarlarını düzelt (RTX 4070 Ti'da hız farkı önemsiz)
        )

        # Ana maske üretici — standart çözünürlük için
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,          # 32×32 = 1024 prompt noktası
            pred_iou_thresh=0.86,        # Tahmin edilen IoU eşiği
            stability_score_thresh=0.92, # Maske stabilitesi eşiği
            min_mask_region_area=50,     # 50px'den küçük tozları yoksay
            box_nms_thresh=0.7,          # Örtüşen kutuları birleştir
        )

        # Küçük nesne taraması için daha hassas ayarlarla ikinci üretici
        self.mask_generator_fine = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=48,          # 48×48 = 2304 nokta (daha yoğun tarama)
            pred_iou_thresh=0.80,        # Daha düşük eşik (küçük nesneler için)
            stability_score_thresh=0.88,
            min_mask_region_area=25,     # Daha küçük nesnelere izin ver
            box_nms_thresh=0.7,
        )

        # autocast durumu — inference sırasında with bloğu ile kullanılır
        self._use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 8
        )

        print(f"[BAŞARILI] SAM 2 belleğe yüklendi. BF16: {'Aktif' if self._use_bf16 else 'Pasif'}")

    def _configure_gpu_optimizations(self):
        """GPU matris çarpımı optimizasyonlarını ayarlar (global, güvenli)."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:  # Ampere (SM 8.0) ve üzeri
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"[OPTİMİZASYON] TF32 aktif — GPU: {props.name}")

    def _run_generator(self, rgb_image, use_fine=False):
        """
        SAM 2 maske üretimini autocast koruması altında çalıştırır.
        
        Args:
            rgb_image: RGB formatlı numpy array
            use_fine: True ise hassas (fine) üreticiyi kullan
            
        Returns:
            list[dict]: SAM 2 maske listesi
        """
        generator = self.mask_generator_fine if use_fine else self.mask_generator

        # autocast'i sadece inference sırasında ve WITH bloğu ile kullanıyoruz
        # Bu sayede context manager düzgün açılıp kapanır (sızıntı yok)
        if self._use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks = generator.generate(rgb_image)
        else:
            masks = generator.generate(rgb_image)

        return masks

    def generate_masks(self, cv2_image, multi_scale=True):
        """
        Görseldeki tüm potansiyel nesneleri metin olmadan bulur.
        
        Çoklu çözünürlük stratejisi:
        1. Orijinal çözünürlükte standart tarama (büyük nesneler)
        2. 2x upscale ile hassas tarama (küçük nesneler)
        3. IoU tabanlı deduplication (çift sayımı önle)
        
        Args:
            cv2_image: BGR formatlı numpy array (OpenCV formatı)
            multi_scale: True ise çoklu çözünürlük kullan (önerilen)
            
        Returns:
            list[dict]: Birleştirilmiş ve deduplike edilmiş maske listesi.
                Her maske dict'i şunları içerir:
                - 'segmentation': bool numpy array (piksel maskesi)
                - 'bbox': [x, y, w, h] (bounding box)
                - 'area': int (piksel alanı)
                - 'predicted_iou': float (model güveni)
                - 'stability_score': float (maske stabilitesi)
        """
        if cv2_image is None or cv2_image.size == 0:
            print("[UYARI] Boş görsel geldi, boş maske listesi döndürülüyor.")
            return []

        # SAM 2 RGB format bekler
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb_image.shape[:2]

        all_masks = []

        # ─── Adım 1: Orijinal çözünürlükte tarama ───
        print(f"[SAM2] Orijinal çözünürlükte taranıyor ({w_orig}x{h_orig})...")
        masks_original = self._run_generator(rgb_image, use_fine=False)
        for mask in masks_original:
            mask['_source'] = 'original'
        all_masks.extend(masks_original)
        print(f"[SAM2] Orijinal: {len(masks_original)} maske bulundu.")

        # ─── Adım 2: 2x upscale ile hassas tarama (küçük nesneler için) ───
        if multi_scale:
            # Upscale boyut sınırı — çok büyük görsellerde VRAM taşmasını önle
            max_upscale_dim = 3200  # Piksel
            scale_factor = 2.0

            if max(h_orig * scale_factor, w_orig * scale_factor) > max_upscale_dim:
                scale_factor = min(max_upscale_dim / h_orig, max_upscale_dim / w_orig)
                print(f"[SAM2] Upscale faktörü VRAM güvenliği için {scale_factor:.2f}x'e düşürüldü.")

            if scale_factor > 1.1:  # Anlamlı bir büyütme varsa devam et
                h_up = int(h_orig * scale_factor)
                w_up = int(w_orig * scale_factor)
                upscaled = cv2.resize(rgb_image, (w_up, h_up), interpolation=cv2.INTER_CUBIC)

                print(f"[SAM2] Upscale çözünürlükte taranıyor ({w_up}x{h_up}, {scale_factor:.1f}x)...")
                masks_upscaled = self._run_generator(upscaled, use_fine=True)

                # Koordinatları orijinal ölçeğe geri dönüştür
                for mask in masks_upscaled:
                    # Bbox'ı ölçekle
                    bx, by, bw, bh = mask['bbox']
                    mask['bbox'] = [
                        bx / scale_factor,
                        by / scale_factor,
                        bw / scale_factor,
                        bh / scale_factor
                    ]
                    # Alanı ölçekle
                    mask['area'] = int(mask['area'] / (scale_factor ** 2))
                    # Segmentation mask'i orijinal boyuta küçült
                    mask['segmentation'] = cv2.resize(
                        mask['segmentation'].astype(np.uint8),
                        (w_orig, h_orig),
                        interpolation=cv2.INTER_NEAREST  # Binary maske için nearest neighbor
                    ).astype(bool)
                    mask['_source'] = 'upscaled'

                all_masks.extend(masks_upscaled)
                print(f"[SAM2] Upscale: {len(masks_upscaled)} maske bulundu.")

        # ─── Adım 3: Duplike maskeleri IoU ile kaldır ───
        deduped = self._deduplicate_masks(all_masks, iou_threshold=0.5)
        print(f"[SAM2] Deduplication sonrası: {len(deduped)} benzersiz maske.")

        return deduped

    def _deduplicate_masks(self, masks, iou_threshold=0.5):
        """
        Örtüşen maskeleri IoU (Intersection over Union) ile birleştirir.
        Aynı nesne hem orijinal hem upscale taramada bulunmuşsa,
        daha yüksek IoU skorlu olanı tutar.
        
        Args:
            masks: SAM 2 maske listesi
            iou_threshold: Bu değerin üzerinde örtüşen maskeler duplike sayılır
            
        Returns:
            list[dict]: Deduplike edilmiş maske listesi
        """
        if len(masks) <= 1:
            return masks

        # Maskeleri predicted_iou skoruna göre azalan sırala (en iyi önce)
        masks_sorted = sorted(masks, key=lambda m: m.get('predicted_iou', 0), reverse=True)
        
        keep = []
        suppressed = set()

        for i, mask_a in enumerate(masks_sorted):
            if i in suppressed:
                continue
            keep.append(mask_a)

            seg_a = mask_a['segmentation']
            area_a = np.sum(seg_a)

            if area_a == 0:
                continue

            for j in range(i + 1, len(masks_sorted)):
                if j in suppressed:
                    continue

                seg_b = masks_sorted[j]['segmentation']
                area_b = np.sum(seg_b)

                if area_b == 0:
                    suppressed.add(j)
                    continue

                # IoU hesapla
                intersection = np.sum(np.logical_and(seg_a, seg_b))
                union = area_a + area_b - intersection

                if union > 0 and (intersection / union) > iou_threshold:
                    suppressed.add(j)

        return keep

    def get_mask_count(self, cv2_image):
        """Hızlı test için: Görselde kaç potansiyel nesne var?"""
        masks = self.generate_masks(cv2_image, multi_scale=False)
        return len(masks)
