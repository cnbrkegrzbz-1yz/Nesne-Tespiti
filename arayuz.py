import os
import sys
import requests
import base64
from dotenv import load_dotenv
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QLineEdit, QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal

load_dotenv()


class SayimGorevi(QThread):
    sonuc_sinyali = pyqtSignal(dict)
    hata_sinyali = pyqtSignal(str)

    def __init__(self, url, ref_yolu, yigin_yolu, nesne_ismi):
        super().__init__()
        self.url = url
        self.ref_yolu = ref_yolu
        self.yigin_yolu = yigin_yolu
        self.nesne_ismi = nesne_ismi

    def run(self):
        try:
            with open(self.ref_yolu, "rb") as f_ref, open(self.yigin_yolu, "rb") as f_yigin:
                files = {'referans_gorsel': f_ref, 'yigin_gorsel': f_yigin}
                data = {'nesne_tanimi': self.nesne_ismi}
                response = requests.post(self.url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                self.sonuc_sinyali.emit(response.json())
            elif response.status_code == 503:
                self.hata_sinyali.emit("Sunucu henüz hazır değil, modeller yükleniyor... (503)")
            else:
                self.hata_sinyali.emit(f"Sunucu {response.status_code} kodu döndürdü.")
                
        except requests.exceptions.Timeout:
            self.hata_sinyali.emit("ZAMAN AŞIMI: İşlem çok uzun sürdü (120sn).")
        except requests.exceptions.ConnectionError:
            self.hata_sinyali.emit(f"BAĞLANTI HATASI: Sunucuya ulaşılamadı. ({self.url})")
        except Exception as e:
            self.hata_sinyali.emit(f"HATA: {str(e)}")



class AkilliSayimArayuzu(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("AnaPencere")
        self.sunucu_url = os.environ.get("SUNUCU_URL", "http://localhost:9090/sayim_yap")
        self.ref_gorsel_yolu = None
        self.yigin_gorsel_yolu = None
        
       
        self.setFont(QFont("Inter", 10)) 
        
        self.arayuzu_kur()

    def arayuzu_kur(self):
        self.setWindowTitle("Akıllı Sayım Sistemi v3 — Premium Mimari")
        self.resize(1100, 850)

        ana_duzen = QVBoxLayout()
        ana_duzen.setContentsMargins(50, 40, 50, 20)

       
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_pixmap = QPixmap("logo.png")
        
        if not logo_pixmap.isNull():
            self.lbl_logo.setPixmap(logo_pixmap.scaled(200, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_logo.setText("KZ MEKATRONİK")
            self.lbl_logo.setObjectName("LogoMetni")
            
       
        logo_glow = QGraphicsDropShadowEffect(self)
        logo_glow.setBlurRadius(30)
        logo_glow.setColor(QColor(59, 130, 246, 90)) 
        logo_glow.setOffset(0, 0)
        self.lbl_logo.setGraphicsEffect(logo_glow)
        
        ana_duzen.addWidget(self.lbl_logo, alignment=Qt.AlignmentFlag.AlignCenter)
        ana_duzen.addSpacing(30)

        
        resimler_duzeni = QHBoxLayout()
        resimler_duzeni.setAlignment(Qt.AlignmentFlag.AlignCenter)
        resimler_duzeni.setSpacing(60)

        
        sol_duzen = QVBoxLayout()
        sol_duzen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_ref_resim = QLabel("📸\n\nReferans Fotoğrafı\nBekleniyor...")
        self.lbl_ref_resim.setObjectName("ResimKutusu")
        self.lbl_ref_resim.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref_resim.setFixedSize(360, 260) 
        self.lbl_ref_resim.setAttribute(Qt.WidgetAttribute.WA_Hover) 
        
        self.btn_ref = QPushButton("Referans Seç")
        self.btn_ref.setObjectName("YariSaydamButon")
        self.btn_ref.setFixedWidth(200)
        self.btn_ref.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ref.clicked.connect(self.ref_sec)
        
        sol_duzen.addWidget(self.lbl_ref_resim)
        sol_duzen.addSpacing(15)
        sol_duzen.addWidget(self.btn_ref, alignment=Qt.AlignmentFlag.AlignCenter)

        
        sag_duzen = QVBoxLayout()
        sag_duzen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_yigin_resim = QLabel("☁️\n\nYığın Fotoğrafı\nBekleniyor...")
        self.lbl_yigin_resim.setObjectName("ResimKutusu")
        self.lbl_yigin_resim.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_yigin_resim.setFixedSize(360, 260)
        self.lbl_yigin_resim.setAttribute(Qt.WidgetAttribute.WA_Hover)
        
        self.btn_yigin = QPushButton("Yığın Fotoğrafı Ekle")
        self.btn_yigin.setObjectName("YariSaydamButon")
        self.btn_yigin.setFixedWidth(200)
        self.btn_yigin.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_yigin.clicked.connect(self.yigin_sec)
        
        sag_duzen.addWidget(self.lbl_yigin_resim)
        sag_duzen.addSpacing(15)
        sag_duzen.addWidget(self.btn_yigin, alignment=Qt.AlignmentFlag.AlignCenter)

        resimler_duzeni.addLayout(sol_duzen)
        resimler_duzeni.addLayout(sag_duzen)

       
        alt_duzen = QVBoxLayout()
        alt_duzen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.entry_nesne = QLineEdit()
        self.entry_nesne.setPlaceholderText("Örn: siyah vida, gri plastik çubuk (opsiyonel)")
        self.entry_nesne.setFixedWidth(550)
        
        self.btn_baslat = QPushButton("SAYIMI BAŞLAT")
        self.btn_baslat.setObjectName("BaslatButonu")
        self.btn_baslat.setFixedWidth(320)
        self.btn_baslat.setFixedHeight(55)
        self.btn_baslat.setCursor(Qt.CursorShape.PointingHandCursor)
        
       
        btn_font = self.btn_baslat.font()
        btn_font.setWeight(QFont.Weight.ExtraBold)
        btn_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        self.btn_baslat.setFont(btn_font)
        self.btn_baslat.clicked.connect(self.sayim_baslat)
        
        
        btn_glow = QGraphicsDropShadowEffect(self)
        btn_glow.setBlurRadius(25)
        btn_glow.setColor(QColor(37, 99, 235, 120)) 
        btn_glow.setOffset(0, 4) 
        self.btn_baslat.setGraphicsEffect(btn_glow)
        
        self.lbl_sonuc = QLabel("")
        self.lbl_sonuc.setObjectName("SonucEtiketi")
        self.lbl_sonuc.setAlignment(Qt.AlignmentFlag.AlignCenter)

        alt_duzen.addWidget(self.entry_nesne, alignment=Qt.AlignmentFlag.AlignCenter)
        alt_duzen.addSpacing(30)
        alt_duzen.addWidget(self.btn_baslat, alignment=Qt.AlignmentFlag.AlignCenter)
        alt_duzen.addSpacing(20)
        alt_duzen.addWidget(self.lbl_sonuc, alignment=Qt.AlignmentFlag.AlignCenter)

        
        ana_duzen.addLayout(resimler_duzeni)
        ana_duzen.addSpacing(40)
        ana_duzen.addLayout(alt_duzen)
        
        ana_duzen.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.setLayout(ana_duzen)

   
    def ref_sec(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(
            self, 
            "Referans Fotoğrafı Seç", 
            "", 
            "Resim Dosyaları (*.jpg *.jpeg *.png *.bmp *.webp);;Tüm Dosyalar (*.*)"
        )
        if dosya_yolu:
            self.ref_gorsel_yolu = dosya_yolu
            pixmap = QPixmap(dosya_yolu)
            self.lbl_ref_resim.setPixmap(pixmap.scaled(self.lbl_ref_resim.width(), self.lbl_ref_resim.height(), Qt.AspectRatioMode.KeepAspectRatioByExpanding))

    def yigin_sec(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(
            self, 
            "Yığın Fotoğrafı Seç", 
            "", 
            "Resim Dosyaları (*.jpg *.jpeg *.png *.bmp *.webp);;Tüm Dosyalar (*.*)"
        )
        if dosya_yolu:
            self.yigin_gorsel_yolu = dosya_yolu
            pixmap = QPixmap(dosya_yolu)
            self.lbl_yigin_resim.setPixmap(pixmap.scaled(self.lbl_yigin_resim.width(), self.lbl_yigin_resim.height(), Qt.AspectRatioMode.KeepAspectRatioByExpanding))

    def sayim_baslat(self):
        nesne_ismi = self.entry_nesne.text().strip()
        if not nesne_ismi: nesne_ismi = "nesne"
            
        if not self.ref_gorsel_yolu or not self.yigin_gorsel_yolu:
            self.lbl_sonuc.setText("Hata: Lütfen referans ve yığın fotoğraflarını eksiksiz yükleyin.")
            self.lbl_sonuc.setStyleSheet("color: rgba(239, 68, 68, 0.9);") 
            return
            
        self.lbl_sonuc.setText("Sistem sayım yapıyor, lütfen bekleyin...\n(Segmentasyon & Eşleştirme aktif)")
        self.lbl_sonuc.setStyleSheet("color: rgba(251, 191, 36, 0.9);") 
        self.btn_baslat.setEnabled(False)
        
        self.islem_thread = SayimGorevi(self.sunucu_url, self.ref_gorsel_yolu, self.yigin_gorsel_yolu, nesne_ismi)
        self.islem_thread.sonuc_sinyali.connect(self.islem_basarili)
        self.islem_thread.hata_sinyali.connect(self.islem_hatali)
        self.islem_thread.start()

    def islem_basarili(self, veri):
        adet = veri.get("adet", 0)
        base64_gorsel = veri.get("sonuc_gorsel_base64", "")
        
        img_data = base64.b64decode(base64_gorsel)
        qimg = QImage.fromData(img_data)
        pixmap = QPixmap.fromImage(qimg)
        
        self.lbl_yigin_resim.setPixmap(pixmap.scaled(self.lbl_yigin_resim.width(), self.lbl_yigin_resim.height(), Qt.AspectRatioMode.KeepAspectRatioByExpanding))
        
        self.lbl_sonuc.setText(f"İşlem Tamamlandı. {adet} adet nesne başarıyla tespit edildi.")
        self.lbl_sonuc.setStyleSheet("color: #10B981; font-weight: bold; font-size: 15px;") 
        self.btn_baslat.setEnabled(True)

    def islem_hatali(self, hata_mesaji):
        self.lbl_sonuc.setText(hata_mesaji)
        self.lbl_sonuc.setStyleSheet("color: rgba(239, 68, 68, 0.9);")
        self.btn_baslat.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    
    global_stil = """
        /* 1. Arka Plan: Merkezden Dışa Radyal Gradyan */
        #AnaPencere {
            background: qradialgradient(cx: 0.5, cy: 0.5, radius: 1, fx: 0.5, fy: 0.5, 
                                        stop: 0 #0f172a, stop: 1 #020617);
        }
        
        /* Alternatif Logo Metni */
        #LogoMetni {
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            letter-spacing: 3px;
        }

        /* 2. Fotoğraf Yükleme Alanları: Glassmorphism */
        #ResimKutusu {
            background-color: rgba(255, 255, 255, 0.03); /* %3 Şeffaflıkta Beyaz */
            color: rgba(255, 255, 255, 0.6); /* Hiyerarşi: %60 Gri/Mavi Metin */
            border: 1px solid rgba(255, 255, 255, 0.1); /* Kesintisiz çok ince yarı saydam çizgi */
            border-radius: 20px; 
            font-size: 14px;
        }
        #ResimKutusu:hover {
            border: 1px solid #3B82F6; /* Logonun canlı mavisi */
            background-color: rgba(255, 255, 255, 0.06); /* Çok hafif aydınlanma */
        }

        /* 3. İkincil Butonlar: Minimal, Şeffaf Zemin */
        #YariSaydamButon {
            background-color: transparent;
            color: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
            font-weight: 600;
        }
        #YariSaydamButon:hover {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid #3B82F6;
            color: #ffffff;
        }

        /* 4. Metin Kutusu: Glassmorphism & Odaklama Halkası */
        QLineEdit {
            background-color: rgba(255, 255, 255, 0.03);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            font-size: 14px;
        }
        QLineEdit:focus {
            border: 1px solid #3B82F6; /* Şık mavi parlama/outline */
            background-color: rgba(59, 130, 246, 0.05);
        }

        /* 5. Ana CTA Butonu: Lineer Geçiş ve Süzülme */
        #BaslatButonu {
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #1D4ED8, stop:1 #3B82F6); 
            color: #FFFFFF;
            border: none;
            border-radius: 12px;
            font-size: 14px;
        }
        #BaslatButonu:hover {
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #1e3a8a, stop:1 #2563EB); 
        }
        #BaslatButonu:disabled {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        #SonucEtiketi {
            font-size: 14px;
        }
    """
    app.setStyleSheet(global_stil)
    
    pencere = AkilliSayimArayuzu()
    pencere.show()
    sys.exit(app.exec())