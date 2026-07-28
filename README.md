# Akıllı Sayım Sistemi

Bir yığın fotoğrafında, verilen bir referans nesne görseline göre kaç adet aynı/benzer nesne olduğunu otomatik sayan görsel sayım sistemi (örn. bir kutu içindeki vida, somun veya parça adedinin sayılması).

## Mimari

- **İstemci** (`arayuz.py`) — PyQt6 tabanlı masaüstü uygulaması
- **Sunucu** (`nesne_tespit/backend/main.py`) — FastAPI backend, "SMC Pipeline" (**S**egment → **M**atch → **C**ount) çalıştırır

## Kullanılan Modeller

- **SAM 2** (Segment Anything Model 2, Meta) — görseldeki tüm nesneleri metin/prompt gerektirmeden, piksel seviyesinde segmente eder. YOLO-World gibi metin prompt'una bağımlı yaklaşımlar yerine tercih edilmiştir, çünkü nesnenin ne olduğunu bilmeye ihtiyaç duymadan "burada bir nesne var" tespiti yapabilir (prompt bağımsız).
- **DINOv2** (Meta) — segmentlenen her nesneyi referans görselle görsel olarak karşılaştırmak için embedding çıkarır, böylece "hangi segmentler referansla eşleşiyor" belirlenir.

## Pipeline

```
reference_processor  → referans görseli işler
segmentor (SAM 2)     → yığın görselindeki tüm nesneleri segmente eder
geometric_filter      → boyut/şekil bazlı alakasız segmentleri eler
visual_matcher (DINOv2) → segmentleri referansla görsel olarak eşleştirir
post_processor        → çakışan tespitleri NMS ile birleştirir, sonucu görselleştirir
```

## API

- `POST /sayim_yap` — `yigin_gorsel`, `referans_gorsel`, `nesne_tanimi` alır; bulunan `adet` ve işaretlenmiş `sonuc_gorsel_base64` döner
- `GET /health` — sunucu ve model yükleme durumunu kontrol eder

## Sistem Gereksinimleri

SAM 2 + DINOv2 aynı anda belleğe yüklendiği ve segmentasyon işlemi ağır olduğu için **GPU pratikte zorunlu** (kod varsayılan olarak `cuda:0` kullanıyor):

- GPU: NVIDIA GPU, en az 8 GB VRAM önerilir (SAM 2 Large checkpoint için); CUDA + cuDNN kurulu olmalı
- RAM: 16 GB+ önerilir
- Disk: checkpoint dosyası için ~1 GB boş alan
- CPU'da da çalışabilir ancak sayım işlemi görsel başına dakikalar sürebilir — pratik kullanım için önerilmez

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
```

SAM 2 checkpoint dosyası (`sam2.1_hiera_large.pt`, ~857 MB) bu depoda yer almaz — [Meta'nın resmi SAM2 deposundan](https://github.com/facebookresearch/sam2) indirip `nesne_tespit/backend/checkpoints/` klasörüne yerleştirin.

Backend'i başlatma:

```bash
cd nesne_tespit/backend
python main.py
```

İstemciyi başlatma:

```bash
python arayuz.py
```

Sunucu adresi kod içine gömülü değildir, ortam değişkeni üzerinden okunur (IP'nin herkese açık depoda görünmemesi için). `.env.example` dosyasını `.env` olarak kopyalayıp kendi sunucu adresinizi girin:

```bash
cp .env.example .env
```

```
SUNUCU_URL=http://<sunucu-ip-adresi>:9090/sayim_yap
```

`.env` dosyası `.gitignore`'da olduğu için GitHub'a asla yüklenmez. Ortam değişkeni ayarlanmazsa kod varsayılan olarak `http://localhost:9090/sayim_yap` adresini kullanır.
