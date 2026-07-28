"""
AŞAMA 5 — Son İşleme (Post-Processing)
========================================
DINOv2 eşleştirmesinden geçen sonuçlara:
1. NMS (Non-Maximum Suppression) uygulayarak çift sayımı önler
2. Annotated görsel oluşturur (bounding box + numara + skor)
"""

from utils.mask_utils import apply_nms
from utils.visualization import draw_results

class PostProcessor:
    """
    Son işleme: NMS + görselleştirme.
    """

    def __init__(self, nms_iou_threshold=0.3, show_scores=True):
        """
        Args:
            nms_iou_threshold: Bu IoU'nun üzerinde örtüşen sonuçlardan birini eler
            show_scores: Annotated görselde similarity skorlarını göster
        """
        self.nms_threshold = nms_iou_threshold
        self.show_scores = show_scores

    def finalize(self, pile_image, matched_results):
        """
        Son işlemleri uygulayarak nihai sayı ve annotated görsel üretir.
        
        Args:
            pile_image: BGR formatlı numpy array (orijinal yığın görseli)
            matched_results: Aşama 4 çıktısı — eşleşen sonuçlar listesi
            
        Returns:
            tuple: (nihai_sonuçlar_listesi, annotated_görsel)
        """
        print(f"[AŞAMA 5] Son işleme başlatılıyor... ({len(matched_results)} eşleşme)")

        # Adım 1: NMS — Çift sayımı önle
        final_results = apply_nms(matched_results, iou_threshold=self.nms_threshold)
        nms_removed = len(matched_results) - len(final_results)
        if nms_removed > 0:
            print(f"  → NMS: {nms_removed} duplike elendi.")

        # Similarity skoruna göre sırala (en yüksek önce)
        final_results.sort(key=lambda r: r['similarity'], reverse=True)

        # Adım 2: Annotated görsel oluştur
        annotated_image = draw_results(
            pile_image, final_results, show_scores=self.show_scores
        )

        print(f"[AŞAMA 5] Sonuç: {len(final_results)} nesne bulundu.")
        return final_results, annotated_image
