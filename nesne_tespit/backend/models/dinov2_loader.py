"""
DINOv2 Model Yükleyici — Görsel DNA Motoru
==========================================
Facebook Research'ün DINOv2 (ViT-B/14) modelini kullanarak
nesnelerin ışıktan bağımsız, semantik görsel embedding'lerini çıkarır.

Neden DINOv2?
- HSV Histogram: Piksel yoğunluk dağılımına bakar → ışık değişince çöker
- SIFT/ORB: Kenar/köşe noktalarına bakar → tekrarlı dokulu endüstriyel parçalarda zayıf
- DINOv2: Nesnenin "ne" olduğunu anlar → ışık, açı, ölçek değişimine dirençli
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

class DINOv2Model:
    """
    DINOv2 ViT-B/14 modeli ile görsel embedding çıkarıcı.
    
    Her görsel için 768 boyutlu, L2-normalize edilmiş bir vektör üretir.
    Bu vektörler arasındaki cosine similarity, iki görselin aynı tür
    nesneyi içerip içermediğini ışıktan bağımsız olarak ölçer.
    """

    def __init__(self, device="cuda:0"):
        """
        Args:
            device: PyTorch cihazı ("cuda:0" veya "cpu")
        """
        print(f"[MODÜL] DINOv2 (ViT-B/14) başlatılıyor... Hedef: {device}")
        self.device = torch.device(device)

        # Facebook'un resmi repo'sundan modeli yükle
        self.model = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_vitb14',
            trust_repo='check'
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Ön-işleme pipeline'ı
        # NOT: CenterCrop yerine doğrudan Resize(224x224) kullanıyoruz.
        # Çünkü yığından kırpılan aday nesneler dikdörtgen bbox'larda gelir,
        # CenterCrop kenarları kesip bilgi kaybına yol açar.
        # DINOv2 hafif aspect ratio değişimine karşı dirençlidir.
        self.transform = T.Compose([
            T.Resize(
                (224, 224), 
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True
            ),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet normalizasyonu
                std=(0.229, 0.224, 0.225)
            ),
        ])

        # Embedding boyutu (ViT-B/14 → 768)
        self.embedding_dim = 768

        print(f"[BAŞARILI] DINOv2 belleğe yüklendi. Embedding boyutu: {self.embedding_dim}")

    def _preprocess(self, cv2_image):
        """
        OpenCV (BGR) görselini DINOv2'nin beklediği tensor formatına dönüştürür.
        
        Args:
            cv2_image: BGR formatlı numpy array
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224) veya None (geçersiz görsel)
        """
        if cv2_image is None or cv2_image.size == 0:
            return None

        # Minimum boyut kontrolü — çok küçük kırpımlar anlamsız embedding üretir
        h, w = cv2_image.shape[:2]
        if h < 8 or w < 8:
            return None

        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Transform uygula ve batch boyutu ekle
        return self.transform(pil_image).unsqueeze(0)

    @torch.no_grad()
    def get_embedding(self, cv2_image):
        """
        Tek bir görselin 768 boyutlu görsel DNA vektörünü çıkarır.
        
        Args:
            cv2_image: BGR formatlı numpy array (OpenCV formatı)
            
        Returns:
            torch.Tensor: L2-normalize edilmiş (1, 768) embedding
            None: Geçersiz görsel durumunda
        """
        tensor = self._preprocess(cv2_image)
        if tensor is None:
            return None

        input_tensor = tensor.to(self.device)
        embedding = self.model(input_tensor)

        # L2 normalizasyon — cosine similarity için gerekli
        return F.normalize(embedding, p=2, dim=1)

    @torch.no_grad()
    def get_embeddings_batch(self, cv2_images, batch_size=32):
        """
        Birden fazla görselin embedding'lerini toplu olarak çıkarır.
        
        Neden batch?
        - Tek tek: 80 aday × ~50ms = ~4 saniye
        - Batch(32): 3 batch × ~150ms = ~0.45 saniye (8-9x hızlanma)
        
        Args:
            cv2_images: BGR formatlı numpy array'lerin listesi
            batch_size: Tek seferde GPU'ya gönderilecek görsel sayısı
                        (16 GB VRAM için 32 güvenli bir değer)
        
        Returns:
            torch.Tensor: (N, 768) boyutlu embedding matrisi
            list: Geçerli görsellerin orijinal indeksleri
                  (geçersiz görseller atlandığı için indeks eşlemesi gerekli)
        """
        all_tensors = []
        valid_indices = []

        # Ön-işleme (CPU'da paralel yapılabilir)
        for idx, img in enumerate(cv2_images):
            tensor = self._preprocess(img)
            if tensor is not None:
                all_tensors.append(tensor.squeeze(0))  # (3, 224, 224)
                valid_indices.append(idx)

        if not all_tensors:
            return torch.empty(0, self.embedding_dim, device=self.device), []

        # Batch'ler halinde GPU'ya gönder
        all_embeddings = []
        for i in range(0, len(all_tensors), batch_size):
            batch_tensors = all_tensors[i:i + batch_size]
            input_batch = torch.stack(batch_tensors).to(self.device)

            embeddings = self.model(input_batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

        # Tüm batch sonuçlarını birleştir
        combined = torch.cat(all_embeddings, dim=0)
        return combined, valid_indices

    @staticmethod
    def cosine_similarity(embedding_a, embedding_b):
        """
        İki embedding arasındaki cosine similarity'yi hesaplar.
        
        Args:
            embedding_a: (1, 768) tensor
            embedding_b: (1, 768) tensor veya (N, 768) tensor
            
        Returns:
            float veya tensor: [-1, 1] aralığında benzerlik skoru
            (1.0 = aynı, 0.0 = ilgisiz, -1.0 = zıt)
        """
        if embedding_a is None or embedding_b is None:
            return 0.0

        # Her iki embedding de L2-normalize edilmiş olduğu için
        # dot product = cosine similarity
        similarity = torch.mm(embedding_a, embedding_b.t())

        # Tek bir skor dönüyorsa float'a çevir
        if similarity.numel() == 1:
            return similarity.item()
        return similarity.squeeze()
