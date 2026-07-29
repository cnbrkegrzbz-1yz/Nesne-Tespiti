"""
Referans Yeterlilik Ön-Kontrolü (VLM)
=======================================
Kullanıcının "öğretim" için yüklediği N referans fotoğrafının, parçanın
ayırt edici açılarını/özelliklerini yeterince kapsayıp kapsamadığını bir
VLM (Ollama üzerinden qwen2.5vl:7b) ile değerlendirir.

Bu modül, ana SMC pipeline'ından (SAM2/DINOv2) TAMAMEN bağımsızdır:
- Farklı bağımlılık (Ollama HTTP client vs. yerel GPU modelleri)
- Farklı yaşam döngüsü (hafif HTTP client, açılışta ağır model yükü yok)
- Farklı hata alanı (ağ/daemon hataları vs. tensor/CV hataları)
Bu yüzden Ollama kapalı/model çekilmemiş olsa bile sayım akışını
ETKİLEMEZ — sadece bu ön-kontrol özelliği bozulur.
"""

import json
import re

import cv2
from ollama import Client


class OllamaBaglantiHatasi(Exception):
    """Ollama'ya ulaşılamadığında veya model bulunamadığında fırlatılır."""
    pass


VLM_SISTEM_PROMPTU = (
    "Sana, bir nesne sayım sistemine bir endüstriyel parçayı öğretmek için "
    "çekilmiş bir referans fotoğraf seti verilecek. Bu fotoğraflar aynı fiziksel "
    "parçanın farklı açılardan çekimleridir.\n\n"
    "Görevin: bu fotoğraf setinin, parçayı yığın içinde güvenilir şekilde "
    "tanıyabilmek için yeterli açı/özellik çeşitliliğini (üstten, yandan, "
    "alttan, ayırt edici işaret/desen vb.) kapsayıp kapsamadığını değerlendirmek.\n\n"
    "Eğer bir 'nesne tanımı' bağlamı verilmişse, sadece o parça tipi için "
    "FİZİKSEL OLARAK ANLAMLI olan açıları iste — örneğin düz bir pulun "
    "'alttan' görünümünü istemek anlamsızdır, simetrik bir vidanın her yönden "
    "aynı görünmesi normaldir.\n\n"
    "SADECE şu JSON formatında çıktı ver, başka hiçbir şey yazma:\n"
    '{"yeterli": true/false, "eksik_yonler": ["somut ve eyleme geçirilebilir '
    'Türkçe öneriler, örn. \\"arkadan bir fotoğraf daha ekleyin\\""], '
    '"aciklama": "kısa genel yorum"}\n\n'
    "eksik_yonler yeterliyse boş liste olmalı. Asla genel/belirsiz bir uyarı "
    "verme ('daha fazla fotoğraf ekleyin' gibi) — her zaman somut bir açı/özellik belirt."
)


class ReferansYeterlilikKontrolcusu:
    """
    Ollama üzerinden çalışan bir VLM (qwen2.5vl:7b) kullanarak referans
    fotoğraf setinin yeterliliğini değerlendirir.
    """

    def __init__(self, host="http://localhost:11434", model="qwen2.5vl:7b", timeout=60):
        """
        Args:
            host: Ollama sunucu adresi
            model: Kullanılacak VLM modeli
            timeout: İstek zaman aşımı (saniye)
        """
        # Güvenlik önlemi: şema eksikse (ör. yanlışlıkla "0.0.0.0:11434" gibi
        # bir değer gelirse) otomatik tamamla — istemci şemasız adrese
        # bağlanmayı denerse sessizce başarısız olur.
        if host and not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        self.host = host
        self.model = model
        self.timeout = timeout

    def degerlendir(self, ref_images, nesne_tanimi=""):
        """
        Referans görsel setini değerlendirir.

        Args:
            ref_images: BGR numpy array listesi (bellekten, diske yazılmadan)
            nesne_tanimi: Opsiyonel kullanıcı açıklaması (VLM'e bağlam olarak verilir)

        Returns:
            dict:
                - 'yeterli': bool
                - 'eksik_yonler': list[str]
                - 'aciklama': str

        Raises:
            OllamaBaglantiHatasi: Ollama'ya ulaşılamadığında veya model
                bulunamadığında — asla sessizce yutulmaz, her zaman çağırana
                yansıtılır.
        """
        print(f"[VLM] {len(ref_images)} referans görseli için yeterlilik kontrolü başlatılıyor...")

        goruntu_baytlari = []
        for img in ref_images:
            basarili, buffer = cv2.imencode('.png', img)
            if basarili:
                goruntu_baytlari.append(buffer.tobytes())

        if not goruntu_baytlari:
            raise ValueError("Değerlendirilecek geçerli görsel yok.")

        baglam_metni = (
            f"Parça açıklaması (kullanıcı tarafından verildi): {nesne_tanimi}\n\n"
            if nesne_tanimi else ""
        )
        kullanici_mesaji = (
            f"{baglam_metni}Ekteki {len(goruntu_baytlari)} referans fotoğrafını "
            "değerlendir ve SADECE istenen JSON formatında yanıt ver."
        )

        try:
            istemci = Client(host=self.host, timeout=self.timeout)
        except Exception as e:
            raise OllamaBaglantiHatasi(f"Ollama istemcisi oluşturulamadı: {e}")

        son_hata = None
        for deneme in range(1, 3):  # 1 deneme + 1 retry
            sicaklik = 0.0 if deneme == 1 else 0.3
            try:
                yanit = istemci.chat(
                    model=self.model,
                    format='json',
                    messages=[
                        {'role': 'system', 'content': VLM_SISTEM_PROMPTU},
                        {
                            'role': 'user',
                            'content': kullanici_mesaji,
                            'images': goruntu_baytlari,
                        },
                    ],
                    options={'temperature': sicaklik, 'num_predict': 512},
                )
            except Exception as e:
                mesaj = str(e).lower()
                if 'not found' in mesaj or 'pull' in mesaj:
                    raise OllamaBaglantiHatasi(
                        f"'{self.model}' modeli bulunamadı — "
                        f"'ollama pull {self.model}' çalıştırın."
                    )
                raise OllamaBaglantiHatasi(
                    f"Ollama sunucusuna ulaşılamadı ({self.host}): {e}"
                )

            ham_metin = yanit['message']['content']
            sonuc, hata = self._json_guvenli_ayristir(ham_metin)

            if hata is None and self._sonuc_gecerli_mi(sonuc):
                print(f"[VLM] Değerlendirme tamamlandı: yeterli={sonuc['yeterli']}")
                return sonuc

            son_hata = hata or "Model çıktısı beklenen alanları içermiyor."
            print(f"[VLM] [UYARI] Deneme {deneme}/2 başarısız: {son_hata}")

        # İki deneme de başarısız — güvenli tarafta kal, kullanıcıyı engelleme
        print(f"[VLM] [UYARI] JSON ayrıştırma tamamen başarısız, güvenli varsayılana dönülüyor.")
        return {
            'yeterli': True,
            'eksik_yonler': [],
            'aciklama': 'VLM değerlendirmesi ayrıştırılamadı, ön-kontrol atlandı.',
        }

    @staticmethod
    def _sonuc_gecerli_mi(sonuc):
        return (
            isinstance(sonuc, dict)
            and isinstance(sonuc.get('yeterli'), bool)
            and isinstance(sonuc.get('eksik_yonler'), list)
        )

    @staticmethod
    def _json_guvenli_ayristir(ham_metin):
        try:
            return json.loads(ham_metin), None
        except json.JSONDecodeError:
            pass

        try:
            eslesme = re.search(r'\{.*\}', ham_metin, re.DOTALL)
            if eslesme:
                return json.loads(eslesme.group()), None
        except json.JSONDecodeError:
            pass

        try:
            ilk = ham_metin.index('{')
            son = ham_metin.rindex('}') + 1
            return json.loads(ham_metin[ilk:son]), None
        except (ValueError, json.JSONDecodeError):
            pass

        return None, "Model çıktısı geçerli JSON formatına dönüştürülemedi."
