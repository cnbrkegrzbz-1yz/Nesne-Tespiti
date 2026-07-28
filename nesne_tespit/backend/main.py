"""
Akıllı Sayım Sistemi — FastAPI Sunucu
======================================
SMC Pipeline v2: Segment → Match → Count

Endpoint: POST /sayim_yap
- yigin_gorsel: Yığın fotoğrafı (File)
- referans_gorsel: Referans fotoğrafı (File)
- nesne_tanimi: Nesne açıklaması (Form, opsiyonel sinyal)

Response:
- adet: Bulunan nesne sayısı
- sonuc_gorsel_base64: Annotated görsel (base64 PNG)

NOT: API imzası eski sunucuyla %100 uyumludur.
     Frontend (arayuz.py) hiç değişmeden çalışır.
"""

import cv2
import numpy as np
import base64
import sys
import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

# Backend kök dizinini Python path'ine ekle
# Bu sayede "from models.xxx import" ve "from pipeline.xxx import" çalışır
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import SayimPipeline

# =====================================================================
# KONFIGURASYON
# =====================================================================
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

# =====================================================================
# UYGULAMA BAŞLATMA
# =====================================================================
app = FastAPI(
    title="Akıllı Sayım Sistemi",
    description="SAM 2 + DINOv2 tabanlı görsel referanslı nesne sayım API'si",
    version="2.0.0"
)

# Pipeline'ı global olarak başlat (modelleri bir kez yükle)
pipeline = None


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


# =====================================================================
# ANA ENDPOINT — Eski API ile %100 uyumlu
# =====================================================================
@app.post("/sayim_yap")
async def sayim_yap(
    yigin_gorsel: UploadFile = File(...),
    referans_gorsel: UploadFile = File(...),
    nesne_tanimi: str = Form(...)
):
    """
    Referanslı akıllı sayım endpoint'i.
    
    Eski sunucu (YOLO + HSV Histogram) ile aynı API imzası.
    Frontend (arayuz.py) hiç değişmeden bu endpoint'e bağlanır.
    """
    global pipeline

    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Pipeline henüz hazır değil, lütfen bekleyin."}
        )

    try:
        # Görselleri oku ve decode et
        ref_bytes = await referans_gorsel.read()
        yigin_bytes = await yigin_gorsel.read()

        img_ref = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_yigin = cv2.imdecode(np.frombuffer(yigin_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img_ref is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Referans görseli okunamadı."}
            )
        if img_yigin is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Yığın görseli okunamadı."}
            )

        # Pipeline'ı çalıştır
        adet, sonuc_img = pipeline.sayim_yap(
            referans=img_ref,
            yigin=img_yigin,
            metin=nesne_tanimi  # Sadece loglama için
        )

        # Sonuç görselini base64'e encode et
        _, buffer = cv2.imencode('.png', sonuc_img)
        base64_gorsel = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "adet": adet,
            "sonuc_gorsel_base64": base64_gorsel
        })

    except Exception as e:
        print(f"[HATA] Sayım sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Sayım hatası: {str(e)}"}
        )


# =====================================================================
# SAĞLIK KONTROLÜ
# =====================================================================
@app.get("/health")
async def health_check():
    """Sunucu ve pipeline durumunu kontrol eder."""
    return {
        "status": "healthy" if pipeline is not None else "loading",
        "pipeline_ready": pipeline is not None,
        "version": "2.0.0 — SMC Pipeline (SAM2 + DINOv2)",
    }


# =====================================================================
# SUNUCU BAŞLATMA
# =====================================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info"
    )
