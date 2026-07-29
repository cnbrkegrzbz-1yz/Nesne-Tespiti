# Akıllı Sayım Sistemi

Bir yığın fotoğrafında, verilen referans nesne görsellerine (birden fazla açıdan öğretilebilir) göre kaç adet aynı/benzer nesne olduğunu otomatik sayan görsel sayım sistemi (örn. bir kutu içindeki vida, somun veya parça adedinin sayılması).

## Mimari

- **Arayüz** (`nesne_tespit/frontend/`) — tarayıcı tabanlı, sunucunun kendisi tarafından servis edilen statik bir web arayüzü (`index.html`/`app.js`/`style.css`). Ek kurulum ya da ayrı bir istemci çalıştırmaya gerek yok — sunucu adresi tarayıcıda açmak yeterli.
- **Sunucu** (`nesne_tespit/backend/main.py`) — FastAPI backend, "SMC Pipeline" (**S**egment → **M**atch → **C**ount) çalıştırır.
- ~~**Eski istemci** (`arayuz.py`)~~ — PyQt6 tabanlı, artık **deprecated**. Çoklu-referans API değişikliğiyle (`referans_gorsel` → `referans_gorseller`) uyumsuz hale geldi, kasıtlı olarak onarılmadı. Yerine web arayüzü kullanılıyor.

## Kullanılan Modeller

- **SAM 2** (Segment Anything Model 2, Meta) — görseldeki tüm nesneleri metin/prompt gerektirmeden, piksel seviyesinde segmente eder. YOLO-World gibi metin prompt'una bağımlı yaklaşımlar yerine tercih edilmiştir, çünkü nesnenin ne olduğunu bilmeye ihtiyaç duymadan "burada bir nesne var" tespiti yapabilir (prompt bağımsız). Çoklu-ölçek (orijinal + upscale) taranır, küçük nesneler de yakalanır.
- **DINOv2** (Meta) — segmentlenen her nesneyi referans görsel(ler)le görsel olarak karşılaştırmak için embedding çıkarır, böylece "hangi segmentler referansla eşleşiyor" belirlenir. Birden fazla referans açısı öğretilmişse, her aday en iyi eşleştiği açıya göre puanlanır.
- **Ollama / qwen2.5vl:7b** (opsiyonel) — referans fotoğraf setinin yeterliliğini (eksik açı var mı, örn. "üstten bir fotoğraf ekleyin") kontrol eden bir VLM ön-değerlendirmesi yapar. Ana sayım pipeline'ından (SAM2/DINOv2) tamamen bağımsızdır; Ollama kapalı/model çekilmemiş olsa bile `/sayim_yap` çalışmaya devam eder, sadece bu tek özellik (`/referans_kontrol`) etkilenir.

## Pipeline

```
reference_processor     → N adet referans görseli işler (elsiz, farklı açılardan)
segmentor (SAM 2)        → yığın görselindeki tüm nesneleri segmente eder (çoklu-ölçek)
geometric_filter         → boyut/şekil bazlı alakasız segmentleri eler (döndürmeye duyarsız,
                            nesne görselde hangi açıda dursa dursun doğru ölçülür)
visual_matcher (DINOv2)   → segmentleri referans(lar)la görsel olarak eşleştirir
post_processor            → çakışan tespitleri NMS ile birleştirir, sonucu görselleştirir
```

## API

- `POST /sayim_yap` — `yigin_gorsel` (tek görsel), `referans_gorseller` (N adet görsel, farklı açılardan öğretim için), `nesne_tanimi` (opsiyonel) alır; bulunan `adet` ve işaretlenmiş `sonuc_gorsel_base64` döner
- `POST /referans_kontrol` — `referans_gorseller`, `nesne_tanimi` (opsiyonel) alır; VLM ile referans setinin yeterliliğini değerlendirir (`yeterli`, `eksik_yonler`, `aciklama`)
- `GET /health` — sunucu ve model yükleme durumunu kontrol eder

## Sistem Gereksinimleri

SAM 2 + DINOv2 aynı anda belleğe yüklendiği ve segmentasyon işlemi ağır olduğu için **GPU pratikte zorunlu** (kod varsayılan olarak `cuda:0` kullanıyor):

- GPU: NVIDIA GPU, en az 8 GB VRAM önerilir (SAM 2 Large checkpoint için); 16 GB, Ollama VLM özelliği de aynı anda kullanılacaksa daha rahat olur. CUDA + cuDNN kurulu olmalı.
- RAM: 16 GB+ önerilir
- Disk: SAM2 checkpoint dosyası için ~1 GB boş alan
- İnternet: `dinov2_loader.py` ilk açılışta (ve model cache'lenmemişse) DINOv2 ağırlıklarını `facebookresearch/dinov2` deposundan çeker — sunucu makinesinin GitHub'a erişimi olmalı
- (Opsiyonel) **Ollama** + `qwen2.5vl:7b` modeli — sadece referans yeterlilik ön-kontrolü (`/referans_kontrol`) için gerekli, ana sayım için gerekmez
- CPU'da da çalışabilir ancak sayım işlemi görsel başına dakikalar sürebilir — pratik kullanım için önerilmez

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
# SAM 2 PyPI'de değildir, ayrıca kurulmalı:
pip install git+https://github.com/facebookresearch/sam2.git
```

SAM 2 checkpoint dosyası (`sam2.1_hiera_large.pt`, ~857 MB) bu depoda yer almaz — [Meta'nın resmi SAM2 deposundan](https://github.com/facebookresearch/sam2) indirip `nesne_tespit/backend/checkpoints/` klasörüne yerleştirin.

(Opsiyonel, VLM referans ön-kontrolü için) Ollama kurup modeli çekin:

```bash
ollama pull qwen2.5vl:7b
```

Backend'i başlatma:

```bash
cd nesne_tespit/backend
python main.py
```

Sunucu `http://0.0.0.0:9090` üzerinde açılır. Arayüze erişmek için tarayıcıda **sunucu adresini doğrudan aç** (ör. `http://localhost:9090` ya da `http://<sunucu-ip>:9090`) — ayrı bir istemci kurmaya, `.env` ayarlamaya gerek yok, arayüz sunucunun kendisi tarafından servis ediliyor.

### Ortam Değişkenleri (opsiyonel, hepsi varsayılan değerle çalışır)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `SAM2_CHECKPOINT` | `checkpoints/sam2.1_hiera_large.pt` | SAM 2 ağırlık dosyası yolu |
| `SAM2_CONFIG` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM 2 konfigürasyon dosyası |
| `DEVICE` | `cuda:0` | PyTorch cihazı (`cuda:0` / `cpu`) |
| `SAYIM_OLLAMA_HOST` | `http://localhost:11434` | Ollama sunucu adresi (VLM ön-kontrolü için) |
| `OLLAMA_VLM_MODEL` | `qwen2.5vl:7b` | VLM ön-kontrolünde kullanılacak model |
| `OLLAMA_TIMEOUT` | `60` | Ollama isteği zaman aşımı (saniye) — model VRAM'den düşmüşse ilk çağrı bunu aşabilir, gerekirse yükseltin |

### Referans Fotoğrafları İçin Pratik Notlar

- Nesneyi **elsiz** çekin (arka plan kaldırma, eli de nesnenin parçası sanıp izole edebiliyor)
- Farklı gerçek açılardan birkaç fotoğraf verin (örn. yandan + üstten) — geometri artık döndürmeye duyarsız olduğu için aynı açının farklı döndürülmüş halini ayrıca çekmenize gerek yok
- VLM ön-kontrolü "eksik açı" uyarısı verse bile, nesnenin şekline göre (ör. simetrik bir çubukta "alttan" görünüm üstten ile aynıdır) bu uyarıyı göz ardı edip devam edebilirsiniz
