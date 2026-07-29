// ---------- Durum pili ----------
const durumNokta = document.getElementById("durum-nokta");
const durumMetin = document.getElementById("durum-metin");
const durumYenileBtn = document.getElementById("durum-yenile-btn");

async function durumKontrolEt() {
  durumNokta.className = "durum-nokta";
  durumMetin.textContent = "Kontrol ediliyor...";
  try {
    const res = await fetch("/health");
    const veri = await res.json();
    const hazir = veri.pipeline_ready === true;
    durumNokta.classList.add(hazir ? "ok" : "hata");
    durumMetin.textContent = hazir ? "Sistem hazır" : "Modeller yükleniyor...";
    return hazir;
  } catch (e) {
    durumNokta.classList.add("hata");
    durumMetin.textContent = "Sunucuya ulaşılamadı";
    return false;
  }
}

durumYenileBtn.addEventListener("click", durumKontrolEt);
durumKontrolEt().then((hazir) => {
  if (!hazir) {
    const aralik = setInterval(async () => {
      const simdiHazir = await durumKontrolEt();
      if (simdiHazir) clearInterval(aralik);
    }, 5000);
  }
});

// ---------- Referans dropzone (çoklu dosya) ----------
const referansInput = document.getElementById("referans-input");
const referansYuklemeAlani = document.getElementById("referans-yukleme-alani");
const referansYuklemeBos = document.getElementById("referans-yukleme-bos");
const thumbnailGrid = document.getElementById("referans-thumbnail-grid");
const vlmDurumu = document.getElementById("referans-vlm-durumu");
const nesneTanimiInput = document.getElementById("nesne-tanimi-input");

let referansDosyalari = [];
let sonVlmSonucu = null; // { yeterli, eksik_yonler, aciklama } | null
let vlmIstekSayaci = 0;
let vlmDebounceZamanlayici = null;

function referansThumbnaillerCiz() {
  thumbnailGrid.innerHTML = "";
  thumbnailGrid.hidden = referansDosyalari.length === 0;
  referansYuklemeBos.hidden = referansDosyalari.length > 0;

  referansDosyalari.forEach((dosya, index) => {
    const item = document.createElement("div");
    item.className = "thumbnail-item";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(dosya);
    img.alt = `Referans ${index + 1}`;
    item.appendChild(img);

    const kaldirBtn = document.createElement("button");
    kaldirBtn.type = "button";
    kaldirBtn.className = "thumbnail-kaldir-btn";
    kaldirBtn.textContent = "×";
    kaldirBtn.setAttribute("aria-label", `Referans ${index + 1}'i kaldır`);
    kaldirBtn.addEventListener("click", () => {
      referansDosyalari.splice(index, 1);
      referansThumbnaillerCiz();
      referansDegistiIsleyici();
    });
    item.appendChild(kaldirBtn);

    thumbnailGrid.appendChild(item);
  });
}

function vlmDurumunuGoster(durum, icerik) {
  vlmDurumu.hidden = false;
  vlmDurumu.className = "vlm-durumu";

  if (durum === "yukleniyor") {
    vlmDurumu.classList.add("yukleniyor");
    vlmDurumu.innerHTML = `<span class="spinner" aria-hidden="true"></span> Referans fotoğraflar değerlendiriliyor...`;
  } else if (durum === "yeterli") {
    vlmDurumu.classList.add("yeterli");
    vlmDurumu.textContent = icerik.aciklama || "Referans fotoğraflar yeterli görünüyor.";
  } else if (durum === "yetersiz") {
    vlmDurumu.classList.add("yetersiz");
    const liste = (icerik.eksik_yonler || [])
      .map((oneri) => `<li>${oneri}</li>`)
      .join("");
    vlmDurumu.innerHTML = `${icerik.aciklama || "Referans fotoğraflar yetersiz görünüyor."}<ul>${liste}</ul>`;
  } else if (durum === "hata") {
    vlmDurumu.classList.add("hata");
    vlmDurumu.textContent = icerik;
  }
}

function vlmDurumunuGizle() {
  vlmDurumu.hidden = true;
  vlmDurumu.innerHTML = "";
}

async function referansYeterlilikKontrolEt() {
  if (referansDosyalari.length === 0) {
    vlmDurumunuGizle();
    sonVlmSonucu = null;
    return;
  }

  const buIstekNo = ++vlmIstekSayaci;
  vlmDurumunuGoster("yukleniyor");

  const formData = new FormData();
  referansDosyalari.forEach((dosya) => formData.append("referans_gorseller", dosya));
  formData.append("nesne_tanimi", nesneTanimiInput.value || "");

  try {
    const res = await fetch("/referans_kontrol", { method: "POST", body: formData });
    const veri = await res.json();

    // Bu istek sonuçlanana kadar referans seti tekrar değiştiyse, eski
    // sonucu görmezden gel — sadece en güncel istek ekrana yazılır.
    if (buIstekNo !== vlmIstekSayaci) return;

    if (!res.ok) {
      sonVlmSonucu = null;
      vlmDurumunuGoster("hata", veri.error || "Referans kontrolü başarısız oldu.");
      return;
    }

    sonVlmSonucu = veri;
    vlmDurumunuGoster(veri.yeterli ? "yeterli" : "yetersiz", veri);
  } catch (e) {
    if (buIstekNo !== vlmIstekSayaci) return;
    sonVlmSonucu = null;
    vlmDurumunuGoster("hata", "Sunucuya ulaşılamadı.");
  }
}

function referansDegistiIsleyici() {
  sayimBaslatButonuGuncelle();
  clearTimeout(vlmDebounceZamanlayici);
  vlmDebounceZamanlayici = setTimeout(referansYeterlilikKontrolEt, 800);
}

referansInput.addEventListener("change", () => {
  referansDosyalari = referansDosyalari.concat(Array.from(referansInput.files));
  referansInput.value = "";
  referansThumbnaillerCiz();
  referansDegistiIsleyici();
});

["dragover", "dragenter"].forEach((evt) => {
  referansYuklemeAlani.addEventListener(evt, (e) => {
    e.preventDefault();
    referansYuklemeAlani.classList.add("surukleniyor");
  });
});

["dragleave", "dragend"].forEach((evt) => {
  referansYuklemeAlani.addEventListener(evt, () => {
    referansYuklemeAlani.classList.remove("surukleniyor");
  });
});

referansYuklemeAlani.addEventListener("drop", (e) => {
  e.preventDefault();
  referansYuklemeAlani.classList.remove("surukleniyor");
  referansDosyalari = referansDosyalari.concat(Array.from(e.dataTransfer.files));
  referansThumbnaillerCiz();
  referansDegistiIsleyici();
});

nesneTanimiInput.addEventListener("input", () => {
  if (referansDosyalari.length > 0) referansDegistiIsleyici();
});

// ---------- Yığın dropzone (tek dosya) ----------
const yiginInput = document.getElementById("yigin-input");
const yiginYuklemeAlani = document.getElementById("yigin-yukleme-alani");
const yiginYuklemeBos = document.getElementById("yigin-yukleme-bos");
const yiginOnizleme = document.getElementById("yigin-onizleme");

let yiginDosyasi = null;

function yiginOnizlemeGoster(dosya) {
  if (!dosya) return;
  yiginDosyasi = dosya;
  yiginOnizleme.src = URL.createObjectURL(dosya);
  yiginOnizleme.hidden = false;
  yiginYuklemeBos.hidden = true;
  sayimBaslatButonuGuncelle();
}

yiginInput.addEventListener("change", () => {
  yiginOnizlemeGoster(yiginInput.files[0]);
});

["dragover", "dragenter"].forEach((evt) => {
  yiginYuklemeAlani.addEventListener(evt, (e) => {
    e.preventDefault();
    yiginYuklemeAlani.classList.add("surukleniyor");
  });
});

["dragleave", "dragend"].forEach((evt) => {
  yiginYuklemeAlani.addEventListener(evt, () => {
    yiginYuklemeAlani.classList.remove("surukleniyor");
  });
});

yiginYuklemeAlani.addEventListener("drop", (e) => {
  e.preventDefault();
  yiginYuklemeAlani.classList.remove("surukleniyor");
  const dosya = e.dataTransfer.files[0];
  if (dosya) {
    yiginInput.files = e.dataTransfer.files;
    yiginOnizlemeGoster(dosya);
  }
});

// ---------- Sayım başlat + onay bandı ----------
const sayimBaslatBtn = document.getElementById("sayim-baslat-btn");
const onayBandi = document.getElementById("onay-bandi");
const onayDevamBtn = document.getElementById("onay-devam-btn");
const onayIptalBtn = document.getElementById("onay-iptal-btn");

let overrideOnaylandi = false;

function sayimBaslatButonuGuncelle() {
  sayimBaslatBtn.disabled = !(referansDosyalari.length > 0 && yiginDosyasi !== null);
  // Referans seti değiştiği için önceki override onayı geçersiz
  overrideOnaylandi = false;
  onayBandi.hidden = true;
}

sayimBaslatBtn.addEventListener("click", () => {
  if (sonVlmSonucu && sonVlmSonucu.yeterli === false && !overrideOnaylandi) {
    onayBandi.hidden = false;
    return;
  }
  sayimiCalistir();
});

onayDevamBtn.addEventListener("click", () => {
  overrideOnaylandi = true;
  onayBandi.hidden = true;
  sayimiCalistir();
});

onayIptalBtn.addEventListener("click", () => {
  onayBandi.hidden = true;
});

// ---------- Sonuç paneli ----------
const sonucBos = document.getElementById("sonuc-bos");
const sonucYukleniyor = document.getElementById("sonuc-yukleniyor");
const sonucHata = document.getElementById("sonuc-hata");
const sonucCevap = document.getElementById("sonuc-cevap");
const sonucAdet = document.getElementById("sonuc-adet");
const sonucGorsel = document.getElementById("sonuc-gorsel");

function sonucDurumunuAyarla(durum, icerik) {
  sonucBos.hidden = durum !== "bos";
  sonucYukleniyor.hidden = durum !== "yukleniyor";
  sonucHata.hidden = durum !== "hata";
  sonucCevap.hidden = durum !== "cevap";

  if (durum === "hata") sonucHata.textContent = icerik;
  if (durum === "cevap") {
    sonucAdet.textContent = `${icerik.adet} adet tespit edilmiştir.`;
    sonucGorsel.src = `data:image/png;base64,${icerik.sonuc_gorsel_base64}`;
  }
}

async function sayimiCalistir() {
  sayimBaslatBtn.disabled = true;
  sonucDurumunuAyarla("yukleniyor");

  const formData = new FormData();
  referansDosyalari.forEach((dosya) => formData.append("referans_gorseller", dosya));
  formData.append("yigin_gorsel", yiginDosyasi);
  formData.append("nesne_tanimi", nesneTanimiInput.value || "");

  try {
    const res = await fetch("/sayim_yap", { method: "POST", body: formData });
    const veri = await res.json();

    if (!res.ok) {
      sonucDurumunuAyarla("hata", veri.error || "Beklenmeyen bir hata oluştu.");
      return;
    }

    sonucDurumunuAyarla("cevap", veri);
  } catch (e) {
    sonucDurumunuAyarla("hata", "Sunucuya ulaşılamadı. Backend'in çalıştığından emin olun.");
  } finally {
    sayimBaslatButonuGuncelle();
  }
}
