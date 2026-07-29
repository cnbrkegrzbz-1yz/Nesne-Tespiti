"""
Akıllı Sayım Sistemi — FastAPI Sunucu
======================================
SMC Pipeline v3: Segment → Match → Count (çoklu-referans + VLM ön-kontrol)

Endpoint: POST /sayim_yap
- yigin_gorsel: Yığın fotoğrafı (File, tek görsel)
- referans_gorseller: Referans fotoğrafları (File, N adet — farklı açılardan öğretim)
- nesne_tanimi: Nesne açıklaması (Form, opsiyonel sinyal)

Endpoint: POST /referans_kontrol
- referans_gorseller: Referans fotoğrafları (File, N adet)
- nesne_tanimi: Nesne açıklaması (Form, opsiyonel — VLM bağlamı için)
Bir VLM (Ollama/qwen2.5vl:7b) ile referans setinin yeterliliğini,
asıl sayım pipeline'ından TAMAMEN bağımsız olarak değerlendirir.

Response (/sayim_yap):
- adet: Bulunan nesne sayısı
- sonuc_gorsel_base64: Annotated görsel (base64 PNG)

NOT: API imzası eski sunucuyla artık uyumlu değil (bilinçli kırılım) —
     referans_gorsel tekil alan, referans_gorseller çoklu alana dönüştü.
     Eski frontend (arayuz.py) bu sürümle çalışmaz, deprecated fallback
     olarak diskte bırakıldı. Yeni arayüz frontend/ altındaki statik
     dosyalardır (bu backend tarafından doğrudan servis edilir).
"""

import cv2
import numpy as np
import base64
import sys
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Backend kök dizinini Python path'ine ekle
# Bu sayede "from models.xxx import" ve "from pipeline.xxx import" çalışır
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import SayimPipeline
from vlm.referans_kontrolcusu import ReferansYeterlilikKontrolcusu, OllamaBaglantiHatasi

# SAM 2 model dosyaları — sunucuda bu yolları kendi kurulumunuza göre güncelleyin
SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "checkpoints/sam2.1_hiera_large.pt"
)
SAM2_CONFIG = os.environ.get(
    "SAM2_CONFIG",
    "configs/sam2.1/sam2.1_hiera_l.yaml"
)
DEVICE = os.environ.get("DEVICE", "cuda:0")

# Referans yeterlilik ön-kontrolü için Ollama/VLM ayarları
# NOT: "OLLAMA_HOST" ismi BİLİNÇLİ olarak kullanılmıyor — Ollama'nın kendisi
# bu ortam değişkenini sunucunun dinleme adresini (ör. 0.0.0.0:11434, LAN
# erişimi için) belirtmek için kullanıyor. Aynı ismi burada da kullanırsak
# istemci yanlışlıkla o değeri hedef sanır ve şema (http://) olmadan
# bağlanmaya çalışıp başarısız olur — bu yüzden farklı bir isim.
SAYIM_OLLAMA_HOST = os.environ.get("SAYIM_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VLM_MODEL = os.environ.get("OLLAMA_VLM_MODEL", "qwen2.5vl:7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

# Kazara aşırı yükleme koruması (referans görseli üst sınırı)
MAKS_REFERANS_SAYISI = 20

app = FastAPI(
    title="Akıllı Sayım Sistemi",
    description="SAM 2 + DINOv2 tabanlı görsel referanslı nesne sayım API'si (çoklu-referans + VLM ön-kontrol)",
    version="3.0.0"
)

# Pipeline'ı global olarak başlat (modelleri bir kez yükle)
pipeline = None

# VLM kontrolcüsü hafif bir HTTP client'tır — model yükü yok, SAM2/DINOv2
# henüz yüklenmemişken bile çalışabilmesi için modül seviyesinde (startup
# dışında) oluşturulur.
vlm_checker = ReferansYeterlilikKontrolcusu(
    host=SAYIM_OLLAMA_HOST,
    model=OLLAMA_VLM_MODEL,
    timeout=OLLAMA_TIMEOUT,
)


@app.on_event("startup")
async def startup_event():
    """Sunucu başlarken modelleri yükle."""
    global pipeline
    try:
        pipeline = SayimPipeline(
            sam2_checkpoint=SAM2_CHECKPOINT,
            sam2_config=SAM2_CONFIG,
            device=DEVICE,
        )
    except Exception as e:
        print(f"[KRİTİK HATA] Pipeline başlatılamadı: {e}")
        print(f"  SAM2 Checkpoint: {SAM2_CHECKPOINT}")
        print(f"  SAM2 Config: {SAM2_CONFIG}")
        print(f"  Device: {DEVICE}")
        raise


def _referans_gorselleri_coz(dosyalar):
    """
    Yüklenen referans dosyalarını cv2 ile çözer.

    Returns:
        tuple: (List[np.ndarray] | None, JSONResponse | None)
            İlki başarılıysa görsel listesi + None, değilse None + hata yanıtı.
    """
    if not dosyalar:
        return None, JSONResponse(
            status_code=400,
            content={"error": "En az bir referans görseli yüklenmelidir."}
        )

    if len(dosyalar) > MAKS_REFERANS_SAYISI:
        return None, JSONResponse(
            status_code=400,
            content={"error": f"En fazla {MAKS_REFERANS_SAYISI} referans görseli yüklenebilir."}
        )

    goruntuler = []
    for i, dosya in enumerate(dosyalar):
        veri = dosya.file.read()
        goruntu = cv2.imdecode(np.frombuffer(veri, np.uint8), cv2.IMREAD_COLOR)
        if goruntu is None:
            return None, JSONResponse(
                status_code=400,
                content={"error": f"Referans görseli #{i + 1} okunamadı."}
            )
        goruntuler.append(goruntu)

    return goruntuler, None


@app.post("/sayim_yap")
def sayim_yap(
    yigin_gorsel: UploadFile = File(...),
    referans_gorseller: List[UploadFile] = File(...),
    nesne_tanimi: str = Form(""),
):
    """
    Çoklu-referanslı akıllı sayım endpoint'i.

    Not: `def` (async değil) — SAM2/DINOv2 çağrıları bloklayan işlemlerdir,
    `async def` içinde çalıştırılırsa event loop'u tüm istekler boyunca
    kilitler. FastAPI/Starlette senkron `def` handler'ları otomatik olarak
    bir thread pool'da çalıştırır.
    """
    global pipeline

    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Pipeline henüz hazır değil, lütfen bekleyin."}
        )

    try:
        img_referanslar, hata_yaniti = _referans_gorselleri_coz(referans_gorseller)
        if hata_yaniti is not None:
            return hata_yaniti

        yigin_bytes = yigin_gorsel.file.read()
        img_yigin = cv2.imdecode(np.frombuffer(yigin_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img_yigin is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Yığın görseli okunamadı."}
            )

        adet, sonuc_img = pipeline.sayim_yap(
            referanslar=img_referanslar,
            yigin=img_yigin,
            metin=nesne_tanimi  # Sadece loglama için
        )

        _, buffer = cv2.imencode('.png', sonuc_img)
        base64_gorsel = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "adet": adet,
            "sonuc_gorsel_base64": base64_gorsel
        })

    except ValueError as e:
        # process_batch: hiçbir referanstan geçerli profil çıkarılamadı
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        print(f"[HATA] Sayım sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Sayım hatası: {str(e)}"}
        )


@app.post("/referans_kontrol")
def referans_kontrol(
    referans_gorseller: List[UploadFile] = File(...),
    nesne_tanimi: str = Form(""),
):
    """
    Referans fotoğraf setinin VLM ile yeterlilik ön-kontrolü.

    Ana sayım pipeline'ından (SAM2/DINOv2) TAMAMEN bağımsızdır — `pipeline`
    global'i henüz yüklenmemiş olsa bile çalışır. Ollama kapalıysa veya
    model çekilmemişse sadece bu endpoint etkilenir, /sayim_yap çalışmaya
    devam eder.
    """
    try:
        img_referanslar, hata_yaniti = _referans_gorselleri_coz(referans_gorseller)
        if hata_yaniti is not None:
            return hata_yaniti

        sonuc = vlm_checker.degerlendir(img_referanslar, nesne_tanimi=nesne_tanimi)
        return JSONResponse(content=sonuc)

    except OllamaBaglantiHatasi as e:
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        print(f"[HATA] Referans kontrolü sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Referans kontrol hatası: {str(e)}"})


@app.get("/health")
async def health_check():
    """Sunucu ve pipeline durumunu kontrol eder."""
    return {
        "status": "healthy" if pipeline is not None else "loading",
        "pipeline_ready": pipeline is not None,
        "version": "3.0.0 — SMC Pipeline v3 (çoklu-referans + VLM ön-kontrol)",
    }


# Statik frontend — /sayim_yap, /referans_kontrol ve /health route'larından
# SONRA tanımlanmalı, aksi halde bu mount "/" altındaki her isteği (API
# dahil) kendine çeker ve API rotaları görünmez olur.
FRONTEND_DIZINI = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIZINI), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info"
    )
