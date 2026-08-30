"""
IMAT 2022-2026 Trend, Kriz, Kontenjan ve Taban Puan Simulatoru (Turkce)
--------------------------------------------------------------------------
Calistirmak icin:  python -m streamlit run imat_simulator_tr.py

Bu panel, dogrulanmis 2022-2025 IMAT verisini (ulusal metrikler +
universite taban puanlari + universite bazi kontenjanlar) onceden
yukler; bilinen 2026 kontenjanlarini isler; ve gecmis verideki en
sert sicramalari kullanarak veriye dayali bir "en kotu senaryo"
projeksiyonu uretir.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="IMAT 2026 Kriz Simulatoru", layout="wide")

# ======================================================================
# 1. DOGRULANMIS TEMEL VERI (buradan degil, arayuzden simulasyon yap)
# ======================================================================

NATIONAL = pd.DataFrame({
    "Yil": [2022, 2023, 2024, 2025, 2026],
    "Sinava_Giren_Toplam": [12226, 10254, 10931, 13495, np.nan],       # 2026 sinavi 29 Eylul'da, henuz bilinmiyor
    "Gecerli_Aday_EU": [np.nan, 3467, 5609, 7202, np.nan],             # 2022 ve 2026 bulunamadi, tahmin edilmedi
    "Ortalama_Puan": [41.8, 24.5, 42.2, 45.1, np.nan],                 # 2026 sonuclari Ekim'de
    "EU_Kontenjan": [718, 921, 1040, 1150, 1319],                      # 2026: DM 1005/6 Agustos 2026 karari
    "NonEU_Kontenjan": [382, 443, 493, 1150, np.nan],                  # 2026 yurt disi ikametli kontenjani bulunamadi
})

UNIVERSITELER = [
    "Bari", "Bologna", "Cagliari", "Catania", "Messina", "Milano Statale",
    "Milano Bicocca", "Napoli Federico II", "Luigi Vanvitelli", "Padova",
    "Padova MedTech", "Parma", "Pavia", "Marche (Ancona)", "La Sapienza",
    "Tor Vergata", "Torino", "Siena (Dis Hekimligi)", "La Sapienza (Dis Hekimligi)",
    "Firenze",
]

# Taban puan sozlugu: universite -> {yil: (EU, NonEU)}  ; None = Yok / henuz bilinmiyor
TABAN_PUANLAR = {
    "Bari":               {2022:(34.1,42.6), 2023:(34.1,31.2), 2024:(55.8,65.8), 2025:(53.1,49.3)},
    "Bologna":            {2022:(41.5,51.5), 2023:(42.6,59.1), 2024:(64.8,74.5), 2025:(60.5,70.1)},
    "Cagliari":           {2022:(None,None), 2023:(None,None), 2024:(54.7,50.9), 2025:(50.9,61.6)},
    "Catania":            {2022:(None,None), 2023:(32.4,35.8), 2024:(54.8,57.7), 2025:(None,61.6)},
    "Messina":            {2022:(33.1,41.0), 2023:(32.8,31.6), 2024:(54.6,61.8), 2025:(51.3,58.2)},
    "Milano Statale":     {2022:(44.3,51.8), 2023:(46.6,60.2), 2024:(67.8,75.3), 2025:(65.8,72.9)},
    "Milano Bicocca":     {2022:(41.0,49.2), 2023:(43.3,54.2), 2024:(64.8,72.7), 2025:(64.5,65.1)},
    "Napoli Federico II": {2022:(37.0,44.4), 2023:(45.2,52.0), 2024:(61.4,68.1), 2025:(58.3,63.1)},
    "Luigi Vanvitelli":   {2022:(34.2,41.9), 2023:(35.0,42.2), 2024:(57.3,63.2), 2025:(52.0,66.2)},
    "Padova":             {2022:(37.9,50.7), 2023:(38.7,49.5), 2024:(63.2,71.6), 2025:(58.6,65.4)},
    "Padova MedTech":     {2022:(None,None), 2023:(None,None), 2024:(None,None), 2025:(None,None)},  # 2026'da ilk kez aciliyor
    "Parma":              {2022:(34.1,43.2), 2023:(35.3,38.7), 2024:(57.6,57.6), 2025:(59.1,67.6)},
    "Pavia":              {2022:(36.7,43.8), 2023:(38.4,53.3), 2024:(59.5,71.5), 2025:(60.1,71.1)},
    "Marche (Ancona)":    {2022:(32.8,41.2), 2023:(35.3,43.0), 2024:(56.5,60.3), 2025:(52.8,58.2)},
    "La Sapienza":        {2022:(42.8,44.6), 2023:(45.4,50.8), 2024:(65.1,73.4), 2025:(62.4,69.1)},
    "Tor Vergata":        {2022:(39.2,44.6), 2023:(38.5,53.4), 2024:(59.5,60.6), 2025:(56.0,69.1)},
    "Torino":             {2022:(37.3,50.1), 2023:(37.4,49.0), 2024:(59.5,70.8), 2025:(59.1,67.1)},
    "Siena (Dis Hekimligi)":       {2022:(32.8,44.9), 2023:(33.2,51.3), 2024:(55.7,69.3), 2025:(46.6,54.8)},
    "La Sapienza (Dis Hekimligi)": {2022:(23.6,46.3), 2023:(38.7,59.1), 2024:(61.8,73.1), 2025:(56.7,71.8)},  # 2025 EU duzeltildi
    "Firenze":            {2022:(None,None), 2023:(None,None), 2024:(None,None), 2025:(None,None)},  # 2026'da ilk kez, akreditasyon bekliyor
}

# Universite bazi KONTENJAN (Seats) gecmisi: universite -> {yil: (EU_koltuk, NonEU_koltuk)}
# Not: Marco Polo koltuklari toplam sayiya dahil edilip yorum olarak ayristirildi.
# Not: Cagliari/Catania icin bazi yillarda koltuk sayisi hicbir kaynakta bulunamadi (None).
KONTENJAN_TARIHSEL = {
    "Bari":               {2022:(42,11),  2023:(69,11),  2024:(69,11),   2025:(69,11),  2026:(89,11)},   # 2024 NonEU = 8+3 Marco Polo
    "Bologna":            {2022:(70,20),  2023:(97,20),  2024:(97,20),   2025:(130,20), 2026:(140,20)},
    "Cagliari":           {2022:(None,None), 2023:(None,None), 2024:(80,20), 2025:(80,20), 2026:(80,18)},
    "Catania":            {2022:(None,None), 2023:(60,None),   2024:(30,30), 2025:(None,60), 2026:(40,40)},
    "Messina":            {2022:(41,42),  2023:(55,56),  2024:(55,56),   2025:(55,56),  2026:(55,61)},
    "Milano Statale":     {2022:(45,25),  2023:(55,25),  2024:(55,15),   2025:(55,15),  2026:(60,15)},
    "Milano Bicocca":     {2022:(26,16),  2023:(30,18),  2024:(30,18),   2025:(30,18),  2026:(40,20)},
    "Napoli Federico II": {2022:(15,25),  2023:(15,25),  2024:(15,25),   2025:(25,45),  2026:(35,45)},
    "Luigi Vanvitelli":   {2022:(50,40),  2023:(60,50),  2024:(60,50),   2025:(60,50),  2026:(70,50)},
    "Padova":             {2022:(51,25),  2023:(80,20),  2024:(75,25),   2025:(75,25),  2026:(75,25)},
    "Padova MedTech":     {2026:(70,15)},  # 2026'da ilk kez aciliyor, akreditasyon bekliyor
    "Parma":              {2022:(60,40),  2023:(75,45),  2024:(75,45),   2025:(75,45),  2026:(80,50)},
    "Pavia":              {2022:(103,40), 2023:(103,40), 2024:(103,40),  2025:(103,35), 2026:(103,40)},  # 2024 NonEU=39+1 Marco Polo; 2025 NonEU=35(+5 olasi yedek)
    "Marche (Ancona)":    {2022:(35,25),  2023:(25,55),  2024:(20,60),   2025:(20,60),  2026:(30,50)},
    "La Sapienza":        {2022:(38,10),  2023:(45,13),  2024:(45,13),   2025:(45,13),  2026:(52,13)},
    "Tor Vergata":        {2022:(25,10),  2023:(40,15),  2024:(40,15),   2025:(60,20),  2026:(65,20)},
    "Torino":             {2022:(70,32),  2023:(70,32),  2024:(70,32),   2025:(70,32),  2026:(73,32)},   # 2022 NonEU=30+2*; 2024 NonEU=31+1 Marco Polo
    "Siena (Dis Hekimligi)":       {2022:(28,15), 2023:(25,12), 2024:(23,12), 2025:(33,2), 2026:(30,8)},
    "La Sapienza (Dis Hekimligi)": {2022:(19,6),  2023:(19,6),  2024:(19,6),  2025:(19,5), 2026:(19,6)},
    "Firenze":            {2026:(50,None)},  # akreditasyon bekliyor, 2026'da ilk kez
}

# Akreditasyonu henuz kesinlesmemis 2026 kontenjanlari - en kotu senaryoda
# bunlarin iptal olabilecegini varsayabilirsin (bkz. Sekme 4).
AKREDITASYON_BEKLEYEN_2026 = {
    "Firenze": 50,
    "Padova MedTech": 70,
    "Napoli Federico II (Veterinaria)": 18,   # ana listeye dahil degil, ayri kontenjan
}

# Bilinen anomaliler / kriz notlari - modelin sessizce gozden kacirmamasi icin
# ayri bir yerde tutuluyor (bkz. Sekme 1).
BILINEN_ANOMALILER = {
    "Catania (EU/yerli kontenjan) - 2025": (
        "2025'te Catania Italya'daki EU/yerli kontenjanini TAMAMEN sifirladi - "
        "sadece yurt disi (non-EU-abroad) icin 60 koltuk acildi, yerli icin 0 koltuk. "
        "2026 kararnamesine gore EU/yerli kontenjani 40 koltukla GERI GELDI. "
        "Sonuc: Catania EU icin 2024->2026 arasinda dogrudan karsilastirilabilir bir "
        "2025 verisi yok - projeksiyon tablolarinda 2024 baz alinarak hesaplaniyor, "
        "ama bu bir 'bosluk yili' sicramasi oldugu icin normal yil-yil sicramalardan "
        "farkli degerlendirilmeli. Kaynak: TestBuddy IMAT 2025-26 kontenjan analizi."
    ),
}

def taban_puan_tablosu():
    satirlar = []
    for uni, yillar in TABAN_PUANLAR.items():
        satir = {"Universite": uni}
        for y, (eu, non) in yillar.items():
            satir[f"{y}_EU"] = eu
            satir[f"{y}_NonEU"] = non
        satirlar.append(satir)
    return pd.DataFrame(satirlar)

def kontenjan_tablosu():
    satirlar = []
    for uni, yillar in KONTENJAN_TARIHSEL.items():
        satir = {"Universite": uni}
        for y in [2022, 2023, 2024, 2025, 2026]:
            eu, non = yillar.get(y, (None, None))
            satir[f"{y}_EU_Koltuk"] = eu
            satir[f"{y}_NonEU_Koltuk"] = non
        satirlar.append(satir)
    return pd.DataFrame(satirlar)

# ======================================================================
# ORTAK TAHMIN FONKSIYONU - Sekme 4, 5 ve 6 BUNU KULLANIR (tek yerden yonetim)
# ======================================================================
def taban_tahmini_hesapla(uni, tur_idx, yontem="ortalama", sensitivite=0.75):
    """
    uni: universite adi
    tur_idx: 0=EU, 1=NonEU
    yontem: "ortalama" (merkezi/tum yillarin ort.), "max" (en kotu senaryo),
            "son_fark" (sadece 2024->2025 farki - Base Case)
    Donen deger: (son_gecerli_yil, son_gecerli_puan, sicrama_degeri, kontenjan_degisim_yuzde, tahmin_2026)
    Herhangi bir veri eksikse ilgili alan(lar) None doner.
    """
    yillar_puan = TABAN_PUANLAR.get(uni, {})
    yillar_koltuk = KONTENJAN_TARIHSEL.get(uni, {})

    degerler = []
    for yil in [2022, 2023, 2024, 2025]:
        cift = yillar_puan.get(yil, (None, None))
        deger = cift[tur_idx]
        if deger is not None:
            degerler.append((yil, deger))

    if len(degerler) < 2:
        return None, None, None, None, None

    farklar = [degerler[i+1][1] - degerler[i][1] for i in range(len(degerler)-1)]
    son_yil, son_puan = degerler[-1]

    if yontem == "max":
        sicrama = max(farklar)
    elif yontem == "son_fark":
        sicrama = farklar[-1]
    else:  # "ortalama"
        sicrama = sum(farklar) / len(farklar)

    koltuk_son_yil = (yillar_koltuk.get(son_yil, (None, None))[tur_idx]
                       if son_yil in yillar_koltuk else None)
    koltuk_2026 = yillar_koltuk.get(2026, (None, None))[tur_idx] if 2026 in yillar_koltuk else None

    if koltuk_son_yil and koltuk_2026 and koltuk_son_yil > 0:
        koltuk_degisim_yuzde = (koltuk_2026 - koltuk_son_yil) / koltuk_son_yil * 100
    else:
        koltuk_degisim_yuzde = None

    baski_etkisi = 0.0
    if koltuk_degisim_yuzde is not None and koltuk_degisim_yuzde < 15:
        baski_etkisi = (15 - koltuk_degisim_yuzde) / 10 * sensitivite

    tahmin_2026 = round(son_puan + sicrama + baski_etkisi, 1)

    return son_yil, son_puan, round(sicrama, 1), (None if koltuk_degisim_yuzde is None else round(koltuk_degisim_yuzde, 1)), tahmin_2026


# 2026 IMAT resmi zaman cizelgesi (Decreto 1005/2026 ve MUR takvimi)
SCORRIMENTO_TAKVIMI_2026 = [
    ("29 Eylul 2026", "Sinav gunu"),
    ("8 Ekim 2026", "Anonim puanlar aciklaniyor (Universitaly)"),
    ("19 Ekim 2026", "Kendi kagidini/puanini gorebiliyorsun"),
    ("26 Ekim 2026", "Isimli ulusal siralama yayinlaniyor (ASSEGNATO / PRENOTATO / IN ATTESA)"),
    ("3 Kasim 2026", "Ilk scorrimento - CINECA bosalan yerleri yeniden dagitiyor"),
    ("Kasim-Aralik 2026", "Ayni ritimle devam: her scorrimento sonrasi 4 is gunu kayit, 5 is gunu 'ilgimi koruyorum' onayi"),
    ("~Ocak-Subat 2027", "Siralama genelde ilk yariyil sonunda kapanir"),
]

# Testbusters'in pozisyon-bazli olasilik kurali (ilk atamadaki SIRA NUMARANA gore, puan degil)
def pozisyon_olasilik_yorumu(sira_farki):
    """sira_farki = son gecerli pozisyon - senin pozisyonun (pozitif ise sen ondeydin)"""
    if sira_farki >= 0:
        if sira_farki <= 50:
            return "COK YUKSEK ihtimal - ilk scorrimento'larda girersin"
        elif sira_farki <= 150:
            return "YUKSEK ihtimal - birkac ay icinde girme sansin yuksek"
        elif sira_farki <= 300:
            return "MUMKUN ama garanti degil - siralamanin sonuna kadar surebilir"
        else:
            return "ZOR - 300+ pozisyon farki gercekci olarak dusuk ihtimal"
    else:
        return "Su an son gecerli pozisyonun disindasin (negatif fark) - sadece ilerideki scorrimentolar seni kurtarabilir"


    return pd.DataFrame(satirlar)

# ======================================================================
# 2. OTURUM DURUMU - sayfa yenilenmeden senaryo ekleyip duzenleyebilirsin
# ======================================================================

if "senaryo_gunlugu" not in st.session_state:
    st.session_state.senaryo_gunlugu = []

st.title("IMAT 2022-2026 Dogrulanmis Veri + Kriz ve Kontenjan Simulatoru")
st.caption("Temel veriler 2022-2025 dogrulama surecinden ve 2026 resmi kontenjan kararindan geliyor. "
           "Sekmelerdeki kontrollerle senaryo uretebilirsin - hicbiri temel veriyi degistirmez.")

sekme1, sekme2, sekme3, sekme4, sekme5, sekme6 = st.tabs([
    "Dogrulanmis Temel Veri", "Senaryo Simulatoru", "Ozel Taban Puan Ekleyici",
    "En Kotu Senaryo (Worst Case)", "Merkezi Tahmin (Ana Senaryo)", "Yerlesim Tahmini (Tercih Listem)"
])

# ----------------------------------------------------------------------
# SEKME 1 - Dogrulanmis temel veri, goz atma
# ----------------------------------------------------------------------
with sekme1:
    st.subheader("Tablo A - Ulusal Metrikler")
    st.dataframe(NATIONAL, use_container_width=True)

    st.subheader("Tablo C - Universite Taban Puanlari (EU / Non-EU)")
    st.dataframe(taban_puan_tablosu(), use_container_width=True)

    st.subheader("Universite Bazi Kontenjan Gecmisi (2022-2026)")
    st.caption("2026 kontenjanlari Decreto Ministeriale n. 1005 (6 Agustos 2026) karariyla resmilesmistir. "
               "Marco Polo ek koltuklari ve olasi yedek koltuklar toplam sayiya dahildir; ayrintilar kod icindeki yorumlarda.")
    st.dataframe(kontenjan_tablosu(), use_container_width=True)

    with st.expander("Akreditasyonu henuz kesinlesmemis 2026 kontenjanlari (risk tasiyor)"):
        for isim, adet in AKREDITASYON_BEKLEYEN_2026.items():
            st.write(f"- **{isim}**: {adet} koltuk - akreditasyon surecinin sonucuna bagli")

    with st.expander("Bilinen anomaliler / kriz notlari (dikkatli okunmali)", expanded=True):
        for baslik, aciklama in BILINEN_ANOMALILER.items():
            st.warning(f"**{baslik}**\n\n{aciklama}")

# ----------------------------------------------------------------------
# SEKME 2 - Genel senaryo simulatoru (EU ve NonEU TAMAMEN AYRI hesaplanir)
# ----------------------------------------------------------------------
with sekme2:
    st.subheader("2026 icin ulusal senaryo simule et")
    st.caption(
        "EU ve Non-EU birbirinden bagimsiz iki havuzdur - ayri kontenjanlari, ayri "
        "rekabet dinamikleri vardir. Bu yuzden asagida ikisi icin AYRI kontenjan "
        "degisimi ve AYRI hassasiyet katsayisi giriyorsun; biri digerini etkilemez. "
        "Sadece 'aday sayisi degisimi' ve 'zorluk degisimi' ortak - cunku ayni sinavi "
        "herkes ayni gun giriyor ve ulusal toplam aday sayisi iki havuzu ayri ayri "
        "raporlamiyor (bu, veri kisitindan kaynaklanan tek ortak varsayimdir)."
    )

    col1, col2 = st.columns(2)
    with col1:
        aday_artisi = st.slider(
            "2025'e gore TOPLAM aday sayisi degisimi (%) - ortak varsayim", -30, 50, 10, step=1
        )
    with col2:
        zorluk_degisimi = st.slider(
            "Algilanan zorluk degisimi (puan, + = daha zor/dusuk puanlar) - ortak varsayim", -10, 10, 0, step=1
        )

    st.markdown("#### EU/Yerli Havuzu Ayarlari")
    col_eu1, col_eu2 = st.columns(2)
    with col_eu1:
        kontenjan_artisi_eu = st.slider(
            "2025'e gore EU kontenjan degisimi (%)", -20, 30, 15, step=1, key="eu_kontenjan"
        )
    with col_eu2:
        hassasiyet_eu = st.slider(
            "EU taban puaninin rekabet endeksine hassasiyeti", 0.0, 10.0, 3.0, step=0.5, key="eu_hassasiyet"
        )

    st.markdown("#### Non-EU Havuzu Ayarlari")
    col_non1, col_non2 = st.columns(2)
    with col_non1:
        kontenjan_artisi_noneu = st.slider(
            "2025'e gore Non-EU kontenjan degisimi (%)", -20, 30, 15, step=1, key="noneu_kontenjan"
        )
    with col_non2:
        hassasiyet_noneu = st.slider(
            "Non-EU taban puaninin rekabet endeksine hassasiyeti", 0.0, 10.0, 3.0, step=0.5, key="noneu_hassasiyet"
        )

    temel_aday = NATIONAL.loc[NATIONAL.Yil == 2025, "Sinava_Giren_Toplam"].values[0]
    temel_kontenjan_eu = NATIONAL.loc[NATIONAL.Yil == 2025, "EU_Kontenjan"].values[0]
    temel_kontenjan_noneu = NATIONAL.loc[NATIONAL.Yil == 2025, "NonEU_Kontenjan"].values[0]
    temel_ortalama = NATIONAL.loc[NATIONAL.Yil == 2025, "Ortalama_Puan"].values[0]

    sim_aday = temel_aday * (1 + aday_artisi / 100)
    sim_ortalama = temel_ortalama - zorluk_degisimi * 0.8

    # --- EU hattı: tamamen bagimsiz ---
    sim_kontenjan_eu = temel_kontenjan_eu * (1 + kontenjan_artisi_eu / 100)
    sim_re_eu = sim_aday / sim_kontenjan_eu
    temel_re_eu = temel_aday / temel_kontenjan_eu
    re_degisim_eu = (sim_re_eu - temel_re_eu) / temel_re_eu
    tahmini_degisim_eu = re_degisim_eu * 10 * hassasiyet_eu / 10

    # --- Non-EU hattı: tamamen bagimsiz, kendi kontenjaniyla ---
    sim_kontenjan_noneu = temel_kontenjan_noneu * (1 + kontenjan_artisi_noneu / 100)
    sim_re_noneu = sim_aday / sim_kontenjan_noneu
    temel_re_noneu = temel_aday / temel_kontenjan_noneu
    re_degisim_noneu = (sim_re_noneu - temel_re_noneu) / temel_re_noneu
    tahmini_degisim_noneu = re_degisim_noneu * 10 * hassasiyet_noneu / 10

    st.markdown("### Simule Edilmis 2026 Sonucu")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**EU/Yerli**")
        st.metric("Tahmini EU Kontenjani", f"{sim_kontenjan_eu:,.0f}", f"{kontenjan_artisi_eu:+d}%")
        st.metric("Tahmini EU Rekabet Endeksi", f"{sim_re_eu:.2f}", f"{sim_re_eu - temel_re_eu:+.2f}")
    with m2:
        st.markdown("**Non-EU**")
        st.metric("Tahmini Non-EU Kontenjani", f"{sim_kontenjan_noneu:,.0f}", f"{kontenjan_artisi_noneu:+d}%")
        st.metric("Tahmini Non-EU Rekabet Endeksi", f"{sim_re_noneu:.2f}", f"{sim_re_noneu - temel_re_noneu:+.2f}")

    st.metric("Tahmini Aday Sayisi (ortak)", f"{sim_aday:,.0f}", f"{aday_artisi:+d}%")
    st.metric("Tahmini Ulusal Ortalama (ortak)", f"{sim_ortalama:.1f}", f"{-zorluk_degisimi*0.8:+.1f}")

    st.markdown("### Tahmini Universite Taban Puan Degisimleri (sezgisel model)")
    st.caption(
        "EU taban puani SADECE EU rekabet endeksindeki degisime gore kayiyor; "
        "Non-EU taban puani SADECE Non-EU rekabet endeksindeki degisime gore kayiyor. "
        "Ikisi birbirini beslemiyor - sadece ortak 'zorluk degisimi' ikisine de ayni sekilde ekleniyor."
    )

    tahmin_satirlari = []
    for uni, yillar in TABAN_PUANLAR.items():
        eu_2025, non_2025 = yillar.get(2025, (None, None))
        tahmin_eu = None if eu_2025 is None else round(eu_2025 + tahmini_degisim_eu - zorluk_degisimi, 1)
        tahmin_non = None if non_2025 is None else round(non_2025 + tahmini_degisim_noneu - zorluk_degisimi, 1)
        tahmin_satirlari.append({"Universite": uni, "2025_EU": eu_2025, "Tahmini_2026_EU": tahmin_eu,
                                  "2025_NonEU": non_2025, "Tahmini_2026_NonEU": tahmin_non})
    st.dataframe(pd.DataFrame(tahmin_satirlari), use_container_width=True)

# ----------------------------------------------------------------------
# SEKME 3 - Universite ve yil bazinda ozel senaryo ekleme
# ----------------------------------------------------------------------
with sekme3:
    st.subheader("Ozel bir taban puan senaryosu ekle")
    st.caption('Ornek: "2026 Bari EU taban puani 2 puan duserse ne olur?"')

    col1, col2, col3 = st.columns(3)
    with col1:
        uni_secim = st.selectbox("Universite", UNIVERSITELER, key="s3_uni")
    with col2:
        kontenjan_secim = st.radio("Kontenjan Turu", ["EU", "Non-EU"], horizontal=True, key="s3_tur")
    with col3:
        temel_yil = st.selectbox("Baz alinacak yil", [2022, 2023, 2024, 2025], index=3, key="s3_yil")

    eu_deger, non_deger = TABAN_PUANLAR.get(uni_secim, {}).get(temel_yil, (None, None))
    temel_deger = eu_deger if kontenjan_secim == "EU" else non_deger

    if temel_deger is None:
        st.warning(f"{uni_secim} icin {temel_yil} yilinda {kontenjan_secim} kaydi yok (Yok) - baska bir kombinasyon sec.")
    else:
        degisim = st.number_input(
            f"{uni_secim} - {kontenjan_secim} - {temel_yil} taban puanina uygulanacak puan degisimi",
            value=-2.0, step=0.5
        )
        yeni_deger = temel_deger + degisim

        st.markdown(f"**{uni_secim} - {kontenjan_secim}, {temel_yil} taban puani: `{temel_deger}` -> "
                    f"Senaryo: `{yeni_deger:.1f}`**")

        if st.button("Bu senaryoyu gunluge ekle"):
            st.session_state.senaryo_gunlugu.append({
                "Universite": uni_secim, "Kontenjan": kontenjan_secim, "Baz Yil": temel_yil,
                "Taban Puan": temel_deger, "Degisim": degisim, "Simule Edilen Deger": round(yeni_deger, 1)
            })

    if st.session_state.senaryo_gunlugu:
        st.markdown("### Senaryo Gunlugu (bu oturum)")
        gunluk_df = pd.DataFrame(st.session_state.senaryo_gunlugu)
        st.dataframe(gunluk_df, use_container_width=True)
        if st.button("Gunlugu temizle"):
            st.session_state.senaryo_gunlugu = []
            st.rerun()

# ----------------------------------------------------------------------
# SEKME 4 - EN KOTU SENARYO: gecmis verideki en sert sicramalari kullanir
# ----------------------------------------------------------------------
with sekme4:
    st.subheader("En Kotu Senaryo (Worst Case) - 2026 Projeksiyonu")
    st.caption(
        "Bu sekme rastgele bir tahmin degil: her universite icin 2022-2025 arasinda "
        "GERCEKTEN yasanmis en sert tek-yillik taban puan sicramasini bulur, 2026'nin "
        "BILINEN kontenjan degisimiyle birlestirir, ve ulusal aday sayisindaki en sert "
        "gecmis buyumeyi (2024->2025: %23.5) 2026'ya da uygulayarak 'her sey ayni anda "
        "en kotu yonde giderse ne olur' sorusuna cevap arar."
    )

    # --- Ulusal en kotu senaryo ---
    toplam_seri = NATIONAL.set_index("Yil")["Sinava_Giren_Toplam"].dropna()
    yillik_buyume = toplam_seri.pct_change().dropna()
    en_sert_buyume = yillik_buyume.max()  # gecmiste yasanan en yuksek YoY aday artisi

    st.markdown("### 1) Ulusal Duzeyde En Kotu Senaryo")
    col1, col2 = st.columns(2)
    with col1:
        akreditasyon_iptal = st.checkbox(
            "Akreditasyon bekleyen kontenjanlar iptal olsun (Firenze 50, Padova MedTech 70, Napoli Vet 18 = toplam 138 koltuk kaybi)",
            value=True
        )
    with col2:
        ekstra_stres = st.slider("Ek stres carpani (gecmis max buyumenin katı olarak)", 1.0, 2.0, 1.0, step=0.1)

    ek_aday_2026 = 13495 * (en_sert_buyume * ekstra_stres)
    worst_aday_2026 = 13495 + ek_aday_2026

    worst_eu_kontenjan_2026 = 1319 - (138 if akreditasyon_iptal else 0)

    worst_rekabet = worst_aday_2026 / worst_eu_kontenjan_2026
    rekabet_2025 = 13495 / 1150

    m1, m2, m3 = st.columns(3)
    m1.metric("En kotu senaryo aday sayisi", f"{worst_aday_2026:,.0f}",
              f"gecmis en sert buyume: %{en_sert_buyume*100*ekstra_stres:.1f}")
    m2.metric("En kotu senaryo EU kontenjani", f"{worst_eu_kontenjan_2026:,.0f}",
              f"-138 akreditasyon riski" if akreditasyon_iptal else "akreditasyon sorunsuz varsayildi")
    m3.metric("En kotu senaryo rekabet endeksi", f"{worst_rekabet:.2f}",
              f"{worst_rekabet - rekabet_2025:+.2f} (2025'e gore)")

    st.divider()

    # --- Universite bazinda en kotu senaryo ---
    st.markdown("### 2) Universite Bazinda En Kotu Taban Puan Projeksiyonu")
    st.caption(
        "Her universite icin 2022-2025 arasindaki ardisik yil farklarindan EN BUYUK "
        "pozitif sicrama bulunur (taban puanin en sert yukseldigi gecis). Buna, 2025->2026 "
        "kontenjan degisiminin baski etkisi eklenir: kontenjan azaldiysa/az arttiysa taban "
        "puan riski daha da yukseklir."
    )

    sensitivite_kontenjan = st.slider(
        "Kontenjan baskisi hassasiyeti (kontenjan %10 daha az artarsa +X puan)",
        0.0, 5.0, 1.5, step=0.25
    )

    worst_case_satirlar = []
    for uni in UNIVERSITELER:
        for tur_idx, tur_adi in enumerate(["EU", "NonEU"]):
            son_yil, son_puan, en_sert_siçrama, koltuk_degisim_yuzde, worst_case_2026 = taban_tahmini_hesapla(
                uni, tur_idx, yontem="max", sensitivite=sensitivite_kontenjan
            )
            bosluk_yili_var = son_yil is not None and son_yil != 2025

            worst_case_satirlar.append({
                "Universite": uni + (" ⚠️" if bosluk_yili_var else ""),
                "Kontenjan Turu": tur_adi,
                "Son Gecerli Yil": son_yil,
                "Son Gecerli Taban Puan": son_puan,
                "Gecmisteki En Sert Siçrama": en_sert_siçrama,
                "Kontenjan Degisimi (%) (son gecerli yil->2026)": koltuk_degisim_yuzde,
                "En Kotu Senaryo 2026 Taban Puan": worst_case_2026,
            })

    worst_df = pd.DataFrame(worst_case_satirlar).dropna(subset=["En Kotu Senaryo 2026 Taban Puan"])
    worst_df = worst_df.sort_values("En Kotu Senaryo 2026 Taban Puan", ascending=False)

    st.caption("⚠️ isareti: bu universitenin 2025 verisi yok (bosluk yili) - projeksiyon son gecerli yildan yapildi, dikkatli yorumla.")
    st.dataframe(worst_df, use_container_width=True, hide_index=True)

    en_riskli = worst_df.head(5)
    if not en_riskli.empty:
        st.markdown("#### En yuksek riskli 5 kombinasyon (en yuksek beklenen taban puan)")
        for _, satir in en_riskli.iterrows():
            st.write(
                f"- **{satir['Universite']} ({satir['Kontenjan Turu']})**: "
                f"{satir['Son Gecerli Yil']}'te {satir['Son Gecerli Taban Puan']} idi, en kotu senaryoda "
                f"**{satir['En Kotu Senaryo 2026 Taban Puan']}** puana kadar cikabilir."
            )

    st.info(
        "Not: Bu model, gecmiste GERCEKTEN yasanmis en sert sicramayi baz alir - uydurma bir "
        "yuzde degil. Ama 2022-2025 sadece 4 veri noktasi oldugu icin, tek bir aykiri yil "
        "(orn. 2023->2024 gecisi) sonucu asiri etkileyebilir. Gercek sinav sonuclari (Ekim 2026) "
        "geldiginde bu model otomatik olarak daha guvenilir hale gelir."
    )

# ----------------------------------------------------------------------
# SEKME 5 - MERKEZI TAHMIN: gecmis TUM yillarin egilimini kullanir (uc deger degil)
# ----------------------------------------------------------------------
with sekme5:
    st.subheader("Merkezi Tahmin (Ana Senaryo) - 2026 Projeksiyonu")
    st.caption(
        "Bu sekme 'en kotu senaryo' degil - onceki TUM yillarin (2022-2025) egilimini "
        "kullanarak en olasi/gercekci 2026 tahminini uretir. Iki farkli yontemi karsilastirir "
        "ve aralarinda agirlik vermeni saglar, cunku tek bir yontem yaniltici olabilir."
    )

    st.markdown("### 1) Ulusal Aday Sayisi - Iki Farkli Merkezi Yontem")

    yillar_np = np.array([2022, 2023, 2024, 2025])
    adaylar_np = np.array([12226, 10254, 10931, 13495])
    egim, sabit = np.polyfit(yillar_np, adaylar_np, 1)
    trend_tahmin_2026 = egim * 2026 + sabit

    buyume_23_24 = (10931 - 10254) / 10254
    buyume_24_25 = (13495 - 10931) / 10931
    ort_momentum = (buyume_23_24 + buyume_24_25) / 2
    momentum_tahmin_2026 = 13495 * (1 + ort_momentum)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Yontem A: Dogrusal Trend (2022-2025 tumu)", f"{trend_tahmin_2026:,.0f}",
                   help="2022'nin yuksek degeri egimi asagi cektigi icin daha temkinli bir tahmin verir.")
    with col2:
        st.metric("Yontem B: Momentum (son 2 yilin ort. buyumesi)", f"{momentum_tahmin_2026:,.0f}",
                   f"ort. buyume: %{ort_momentum*100:.1f}",
                   help="Sadece 2023->2024 ve 2024->2025 buyume hizlarinin ortalamasini kullanir, 2022'yi disarida birakir.")

    agirlik = st.slider(
        "Iki yontem arasi agirlik (0 = tamamen Trend, 1 = tamamen Momentum)",
        0.0, 1.0, 0.5, step=0.05
    )
    merkezi_aday_2026 = trend_tahmin_2026 * (1 - agirlik) + momentum_tahmin_2026 * agirlik

    st.markdown(f"### Secilen Agirlikla Merkezi Tahmin: **{merkezi_aday_2026:,.0f}** aday")

    merkezi_kontenjan_2026 = 1319  # bilinen gercek deger
    merkezi_ci_2026 = merkezi_aday_2026 / merkezi_kontenjan_2026
    ci_2025 = 13495 / 1150

    m1, m2, m3 = st.columns(3)
    m1.metric("2026 EU Kontenjani (bilinen, gercek)", f"{merkezi_kontenjan_2026:,}")
    m2.metric("Merkezi Rekabet Endeksi 2026", f"{merkezi_ci_2026:.2f}", f"{merkezi_ci_2026 - ci_2025:+.2f} (2025'e gore)")
    m3.metric("2025 Rekabet Endeksi (karsilastirma)", f"{ci_2025:.2f}")

    st.divider()

    st.markdown("### 2) Universite Bazinda Merkezi Taban Puan Tahmini")
    st.caption(
        "Her universite icin 2022-2025 arasindaki ardisik yil farklarinin ORTALAMASI alinir "
        "(en kotu senaryodaki gibi maksimum degil). Buna, 2025->2026 arasi bilinen gercek "
        "kontenjan degisiminin orta duzey bir baski katsayisiyla etkisi eklenir."
    )

    merkezi_sensitivite = st.slider(
        "Kontenjan baskisi hassasiyeti (merkezi senaryo icin, en kotu senaryodan daha dusuk tutulmali)",
        0.0, 3.0, 0.75, step=0.25
    )

    merkezi_satirlar_eu = []
    merkezi_satirlar_noneu = []
    for uni in UNIVERSITELER:
        for tur_idx, tur_adi in enumerate(["EU", "NonEU"]):
            son_yil, son_puan, ort_degisim, koltuk_degisim_yuzde, merkezi_2026 = taban_tahmini_hesapla(
                uni, tur_idx, yontem="ortalama", sensitivite=merkezi_sensitivite
            )
            bosluk_yili_var = son_yil is not None and son_yil != 2025

            satir = {
                "Universite": uni + (" ⚠️" if bosluk_yili_var else ""),
                "Son Gecerli Yil": son_yil,
                "Son Gecerli Taban Puan": son_puan,
                "Ortalama Yillik Degisim": ort_degisim,
                "Kontenjan Degisimi (%) (son gecerli yil->2026)": koltuk_degisim_yuzde,
                "Merkezi Tahmin 2026 Taban Puan": merkezi_2026,
            }

            if tur_idx == 0:
                merkezi_satirlar_eu.append(satir)
            else:
                merkezi_satirlar_noneu.append(satir)

    merkezi_df_eu = pd.DataFrame(merkezi_satirlar_eu).dropna(subset=["Merkezi Tahmin 2026 Taban Puan"])
    merkezi_df_eu = merkezi_df_eu.sort_values("Merkezi Tahmin 2026 Taban Puan", ascending=False)

    merkezi_df_noneu = pd.DataFrame(merkezi_satirlar_noneu).dropna(subset=["Merkezi Tahmin 2026 Taban Puan"])
    merkezi_df_noneu = merkezi_df_noneu.sort_values("Merkezi Tahmin 2026 Taban Puan", ascending=False)

    st.caption("⚠️ isareti: bu universitenin 2025 verisi yok (bosluk yili) - projeksiyon son gecerli yildan yapildi, dikkatli yorumla.")

    st.markdown("#### EU / Yerli Havuzu - Merkezi Tahmin")
    st.dataframe(merkezi_df_eu, use_container_width=True, hide_index=True)

    st.markdown("#### Non-EU Havuzu - Merkezi Tahmin")
    st.dataframe(merkezi_df_noneu, use_container_width=True, hide_index=True)

    st.info(
        "Bu tahminler 4 veri noktasindan (2022-2025) turetildigi icin istatistiksel olarak "
        "zayif bir temele sahiptir - ozellikle 2023 (Cambridge'den MUR'a gecis yili) bir "
        "aykiri deger olabilir ve ortalamayi carpitiyor olabilir. Istersen 2023'u disarida "
        "birakip sadece 2024-2025 farkini kullanan alternatif bir hesaplama da ekleyebilirim."
    )

# ----------------------------------------------------------------------
# SEKME 6 - YERLESIM TAHMINI: senin puanin + tercih listen -> 3 senaryo
# ----------------------------------------------------------------------
with sekme6:
    st.subheader("Kendi Puanin ve Tercih Listenle Yerlesim Tahmini")
    st.caption(
        "Puanini gir, tercih listeni SIRAYLA sec (ilk sectigin = 1. tercihin). "
        "Uc ayri model (Base Case / Merkezi / En Kotu) ayni anda hesaplanir ve "
        "tercih sirana gore ilk 'girersin' dedigi okul gosterilir."
    )

    col1, col2 = st.columns(2)
    with col1:
        senin_puanin = st.number_input("IMAT Puanin (2025 veya beklenen 2026 puani)", value=50.1, step=0.1)
    with col2:
        kontenjan_turu_s6 = st.radio("Kontenjan Turun", ["EU", "NonEU"], horizontal=True, key="s6_tur")

    tur_idx_s6 = 0 if kontenjan_turu_s6 == "EU" else 1

    tercih_listesi = st.multiselect(
        "Tercih Listeni SIRAYLA sec (ilk tikladigin = en yuksek tercihin)",
        UNIVERSITELER,
        default=["La Sapienza", "Milano Statale", "Milano Bicocca", "Bologna", "Pavia",
                 "Padova", "Torino", "Tor Vergata", "Napoli Federico II", "Luigi Vanvitelli",
                 "Bari", "Cagliari", "Catania", "Firenze", "Messina", "Padova MedTech",
                 "La Sapienza (Dis Hekimligi)", "Siena (Dis Hekimligi)"]
    )

    if tercih_listesi:
        sonuc_satirlari = []
        for sira, uni in enumerate(tercih_listesi, start=1):
            son_yil_bc, son_puan_bc, sicrama_bc, koltuk_bc, tahmin_bc = taban_tahmini_hesapla(
                uni, tur_idx_s6, yontem="son_fark", sensitivite=0.75
            )
            _, _, _, _, tahmin_merkezi = taban_tahmini_hesapla(uni, tur_idx_s6, yontem="ortalama", sensitivite=0.75)
            _, _, _, _, tahmin_worst = taban_tahmini_hesapla(uni, tur_idx_s6, yontem="max", sensitivite=1.5)

            if tahmin_bc is None:
                sonuc_satirlari.append({
                    "Sira": sira, "Universite": uni, "Son Yil": None, "Son Puan": None,
                    "Base Case 2026": None, "Merkezi 2026": None, "En Kotu 2026": None,
                    "Fark (Base Case)": None, "Durum": "VERI YOK / yeni kontenjan",
                })
                continue

            fark_bc = round(senin_puanin - tahmin_bc, 1)
            durum = "GECER" if fark_bc >= 0 else "ACIK VAR"

            sonuc_satirlari.append({
                "Sira": sira, "Universite": uni, "Son Yil": son_yil_bc, "Son Puan": son_puan_bc,
                "Base Case 2026": tahmin_bc, "Merkezi 2026": tahmin_merkezi, "En Kotu 2026": tahmin_worst,
                "Fark (Base Case)": fark_bc, "Durum": durum,
            })

        sonuc_df = pd.DataFrame(sonuc_satirlari)
        st.dataframe(sonuc_df, use_container_width=True, hide_index=True)

        gecerler = sonuc_df[sonuc_df["Durum"] == "GECER"]
        if not gecerler.empty:
            ilk_gecen = gecerler.iloc[0]
            st.success(
                f"Tercih sirana gore (Base Case modeline gore) ilk girecegin yer: "
                f"**#{ilk_gecen['Sira']} - {ilk_gecen['Universite']}** "
                f"(marj: {ilk_gecen['Fark (Base Case)']:+.1f} puan)"
            )
        else:
            st.warning("Base Case modeline gore tercih listendeki hicbir okula girme ihtimalin yuksek gorunmuyor.")

    st.divider()
    st.markdown("### 2026 Scorrimento (Yerlesim) Zaman Cizelgesi")
    st.caption("Bunlar tahmin degil - MUR'un Decreto 1005/2026 ile ilan ettigi kesin tarihler.")
    for tarih, aciklama in SCORRIMENTO_TAKVIMI_2026:
        st.write(f"- **{tarih}**: {aciklama}")

    st.divider()
    st.markdown("### Pozisyon-Bazli Olasilik Hesaplayici")
    st.caption(
        "Bu, PUAN degil SIRA NUMARANI kullanir. Isimli siralama (26 Ekim 2026) "
        "ciktiktan sonra, ilgilendigin okulun ilk atamadaki son gecerli pozisyonuyla "
        "kendi pozisyonunu karsilastir."
    )
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        son_gecerli_pozisyon = st.number_input("Okulun ilk atamadaki son gecerli pozisyonu", min_value=0, value=100, step=1)
    with col_p2:
        senin_pozisyonun = st.number_input("Senin siralamadaki pozisyonun", min_value=0, value=120, step=1)

    fark_pozisyon = son_gecerli_pozisyon - senin_pozisyonun
    yorum = pozisyon_olasilik_yorumu(fark_pozisyon)
    st.info(f"Pozisyon farki: {fark_pozisyon:+d} -> **{yorum}**")

st.divider()
st.caption("Temel veri, 2022-2025 IMAT capraz kaynak dogrulamasindan alinmistir (Testbusters, Locomotive, Futura, "
           "Wauniversity). 2024/2025 EU gecerli aday sayilari (5.609 / 7.202) ve taban puanlar kullanicinin "
           "verdigi kaynaklara gore sabitlenmistir. 2026 kontenjanlari Decreto Ministeriale n. 1005 (6 Agustos 2026) "
           "kararindan alinmistir. 2022 EU gecerli aday sayisi ve 2026 sinav sonuclari henuz bilinmiyor.")
