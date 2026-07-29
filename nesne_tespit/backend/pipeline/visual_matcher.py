"""
AŞAMA 4 — Görsel Eşleştirme (Visual Matching)
===============================================
Geometrik filtreden geçen her aday maskeyi DINOv2 embedding'i ile
referans nesneye karşı görsel olarak karşılaştırır.

Bu aşama, HSV histogram'ın yapamadığını yapar:
- Gölgede kalan nesneleri DOĞRU eşleştirir (ışığa dirençli)
- Farklı nesne türlerini ayırt eder (semantik anlam)

Dinamik Eşik Stratejisi:
- Sabit eşik yerine, Otsu yöntemiyle skor dağılımını iki kümeye ayırır
  (kümeler-arası varyansı maksimize eden ayrım noktasını bulur)
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

    def match(self, pile_image, candidate_masks, ref_embeddings):
        """
        Aday maskeleri referans embedding(ler) ile karşılaştırır.

        İşlem:
        1. Her aday maskeyi pile_image'dan kırp
        2. Batch olarak DINOv2 embedding çıkar
        3. Cosine similarity hesapla (her aday için TÜM referanslara karşı)
        4. Referanslar arası maksimumu al (her aday, öğretilen açılardan
           EN İYİ eşleştiği açıya göre puanlanır)
        5. Dinamik eşik belirle
        6. Eşiğin üzerindeki adayları döndür

        Args:
            pile_image: BGR formatlı numpy array (yığın fotoğrafı)
            candidate_masks: Filtrelenmiş maske listesi (Aşama 3 çıktısı)
            ref_embeddings: Referans DINOv2 embedding'leri, (K, 768) — K adet
                öğretilen açının embedding'i üst üste yığılmış (Aşama 1 çıktısı)

        Returns:
            list[dict]: Eşleşen sonuçlar, her biri:
                - 'mask': SAM 2 mask dict
                - 'similarity': float (cosine similarity skoru, K referans üzerinden max)
        """
        print(f"[AŞAMA 4] Görsel eşleştirme başlatılıyor... ({len(candidate_masks)} aday)")

        if not candidate_masks or ref_embeddings is None or ref_embeddings.shape[0] == 0:
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
        # ref_embeddings: (K, 768), candidate_embeddings: (N, 768) -> (K, N)
        # Adım 3b: Her aday için K referans arasından en iyi (max) skoru al.
        # Neden max (mean/min değil): bir aday, yığında çekildiği TEK açıya en
        # yakın referans açısıyla eşleşir; diğer öğretilen açılarla yüksek
        # skor üretmesi için hiçbir neden yok. mean, öğretilen açı sayısı
        # arttıkça gerçek eşleşmelerin skorunu yapay olarak düşürür (yanlış
        # yönde); min ise tek bir kötü açı referansının tüm eşleşmeyi veto
        # etmesine izin verir. max, kaç açı öğretildiğinden bağımsız doğru
        # davranan tek agregasyondur.
        similarity_matrix = torch.mm(ref_embeddings, candidate_embeddings.t())  # (K, N)
        similarities, _ = similarity_matrix.max(dim=0)  # (N,)
        similarities = similarities.cpu().numpy()

        print(f"  → Similarity aralığı: [{similarities.min():.3f}, {similarities.max():.3f}]")

        # Debug: her adayın skorunu ve konumunu tek tek yazdır (skora göre
        # azalan sırayla) — hangi adayların neden elendiğini pozisyonlarıyla
        # birlikte görebilmek için. Kalıcı, ucuz bir log satırı.
        debug_sirali = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )
        for i in debug_sirali:
            crop_idx = embed_valid_indices[i]
            mask_idx = valid_mask_indices[crop_idx]
            bbox = candidate_masks[mask_idx]['bbox']
            props = candidate_masks[mask_idx].get('_geometric_props', {})
            ar = props.get('aspect_ratio')
            ar_str = f"{ar:.2f}" if ar is not None else "?"
            print(f"    Aday: skor={similarities[i]:.3f} bbox={bbox} AR={ar_str}")

        # Adım 4: Dinamik eşik belirle
        threshold = self._determine_dynamic_threshold(similarities)
        print(f"  → Dinamik eşik: {threshold:.3f}")

        # Adım 5: Eşleşenleri topla
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
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
        Sabit eşik döndürür (min_threshold).

        NOT (eskiden sırayla "en büyük tek komşu boşluk" ve ardından Otsu
        denendi — ikisi de terk edildi): gerçek testlerde (çubuklar VE
        vidalar) aynı nesnenin farklı kopyalarının skorları doğal olarak
        geniş bir aralığa yayılabiliyor (ör. 16 özdeş vida için
        similarity 0.390-0.804 arasına yayıldı), AMA çöp/alakasız
        adayların skoru da HER testte güvenilir şekilde düşük kaldı
        (~0.20-0.35 altı). Yani gerçek/sahte arasındaki asıl sınır zaten
        net ve büyük (vidalarda 0.187'lik boşluk) — sorun sınırı BULMAK
        değil, "en büyük boşluğu" ararken bu net sınırı değil, gerçek
        nesnelerin KENDİ İÇİNDEKİ daha küçük bir dalgalanmayı seçmekti
        (hem gap-yöntemi hem Otsu bu tuzağa düştü, çubuklarda 10'dan
        2'ye, vidalarda 16'dan 14'e düşürdü). v1'in (çoklu-referans/VLM
        öncesi) tek sabit eşikle 16/16 bulması da bunu doğruluyor.
        Çözüm: karmaşık "dinamik" mantığı tamamen bırakıp, veriyle zaten
        güvenilir olduğu kanıtlanmış tek bir sabit tabanı kullanmak.

        Args:
            scores: numpy array — cosine similarity skorları (kullanılmıyor,
                imza geriye dönük uyumluluk için korundu)

        Returns:
            float: Eşik değeri (min_threshold)
        """
        return self.min_threshold
