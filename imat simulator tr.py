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
import math
from scipy.stats import qmc, norm as norm_dagilim, t as t_dagilim, binom as binom_dagilim

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
    # PUANLAMA TAVANI: 60 soru x (+1.5 dogru) = 90 mutlak matematiksel tavan.
    # Hicbir tahmin bu sinirin uzerine cikamaz; asagi yonde de 0'in altina inemez.
    tahmin_2026 = max(0.0, min(90.0, tahmin_2026))

    return son_yil, son_puan, round(sicrama, 1), (None if koltuk_degisim_yuzde is None else round(koltuk_degisim_yuzde, 1)), tahmin_2026


def _farklar_bul(uni, tur_idx, gecis_yili_haric=True):
    """
    Bir universitenin 2022-2025 arasindaki ardisik yil-yil taban puan farklarini dondurur.
    gecis_yili_haric=True (varsayilan): 2023->2024 farki HARIC TUTULUR. Sebep: bu, TUM
    okullarda ayni anda yasanmis, ~+21/+22 puanlik cok buyuk, tek seferlik bir SISTEMIK
    sicrama (muhtemelen Cambridge'den MUR'a gecis yili - kodun 5. sekmesindeki notta zaten
    isaret ediliyor). Bu bir 'normal yillik oynaklik' degil, 'yapisal kirilma'dir - 2026'da
    tekrarlanmasi beklenmez. Havuzlama (pooling) bunu KUCULTEMEZ cunku her okulda ayni sekilde
    var - bu yuzden dogrudan disarida birakmak gerekiyor.
    """
    yillar_puan = TABAN_PUANLAR.get(uni, {})
    degerler = []
    for yil in [2022, 2023, 2024, 2025]:
        cift = yillar_puan.get(yil, (None, None))
        deger = cift[tur_idx]
        if deger is not None:
            degerler.append((yil, deger))
    if len(degerler) < 2:
        return []
    farklar = []
    for i in range(len(degerler) - 1):
        yil_onceki, yil_sonraki = degerler[i][0], degerler[i + 1][0]
        if gecis_yili_haric and yil_onceki == 2023 and yil_sonraki == 2024:
            continue
        farklar.append(degerler[i + 1][1] - degerler[i][1])
    return farklar


def havuzlanmis_varyans(tur_idx, gecis_yili_haric=True):
    """
    KLASIK POOLED VARIANCE (ANOVA'da/t-testlerde kullanilan standart formul):
    sum( (n_i-1) * var_i ) / sum( n_i-1 )
    Her universitenin kendi farklarindan hesaplanan varyansi, o okulun serbestlik
    derecesiyle (df_i = n_i-1) agirliklandirip TUM okullar icin ortak/tipik bir
    "temel volatilite" uretir.
    """
    pay, payda = 0.0, 0
    hepsi = []
    for uni in UNIVERSITELER:
        farklar = _farklar_bul(uni, tur_idx, gecis_yili_haric=gecis_yili_haric)
        hepsi.extend(farklar)
        if len(farklar) >= 2:
            var_i = float(np.var(farklar, ddof=1))
            df_i = len(farklar) - 1
            pay += df_i * var_i
            payda += df_i
    if payda == 0:
        return float(np.var(hepsi)) if hepsi else 1.0
    return pay / payda


def volatilite_index_hesapla(uni, tur_idx, k=3, gecis_yili_haric=True):
    """
    SEFFAF VOLATILITE HESABI - shrinkage/partial pooling (Efron-Morris / James-Stein
    tipi tahminci - kucuk orneklemli gruplarda standart bir istatistik teknigi):

        shrunk_var = (df_i * kendi_varyansi + k * havuzlanmis_varyans) / (df_i + k)

    df_i = o okulun kac yil-yil farki oldugu (genelde SADECE 2 veya 3 - cok az).
    k = havuza verilen agirlik (varsayilan 3).

    NOT: gecis_yili_haric=True iken 2023->2024 sistemik sicramasi ZATEN cikarilmis
    oluyor - bu yuzden havuzlanmis varyans artik daha kucuk/gercekci cikar. Bazi
    okullarda (Cagliari, Catania, Bari gibi bosluk-yili olanlarda) bu cikarma
    sonrasi elde SADECE 0 veya 1 fark kalabilir - bu durumda okulun kendi
    varyansi hic hesaplanamaz, tamamen havuza guvenilir (asagida ele alinir).
    """
    farklar = _farklar_bul(uni, tur_idx, gecis_yili_haric=gecis_yili_haric)
    pooled_var = havuzlanmis_varyans(tur_idx, gecis_yili_haric=gecis_yili_haric)
    if not farklar:
        return float(np.sqrt(max(pooled_var, 0.0)))
    df_i = len(farklar) - 1
    if df_i >= 1:
        individual_var = float(np.var(farklar, ddof=1))
        shrunk_var = (df_i * individual_var + k * pooled_var) / (df_i + k)
    else:
        shrunk_var = pooled_var  # tek fark var (ya da hic yok), kendi varyansi hesaplanamaz -> tamamen havuza guven
    return float(np.sqrt(max(shrunk_var, 0.0)))


GUVENILIRLIK_ESIGI = 89.5  # bu deger veya uzerine cikan %95 tahmini "GUVENILMEZ" olarak isaretlenir


def taban_ornek_uret(n_sim, seed, boyut=2, ornekleme_yontemi="srs"):
    """
    [0,1) araliginda boyut-sutunlu N ornek uretir - 3 farkli yontemle:
      - 'srs' (Basit Rastgele): her nokta tamamen bagimsiz rastgele - klasik Monte Carlo
      - 'lhs' (Katmanli / Latin Hypercube): [0,1) araligi n_sim esit dilime bolunur, HER
        dilimden TAM OLARAK bir ornek alinir - hicbir bolge atlanmaz, daha duzgun kapsama
      - 'qmc' (Quasi-Monte Carlo / Sobol dizisi): tamamen rastgele degil, matematiksel
        olarak en es-dagilmis (dusuk dispersiyonlu) deterministik bir dizi - ayni n_sim
        icin SRS'den daha hizli/duzgun yakinsar
    Donen: (n_sim, boyut) sekilli array.
    """
    rng = np.random.default_rng(seed)
    if ornekleme_yontemi == "lhs":
        sampler = qmc.LatinHypercube(d=boyut, seed=seed)
        return sampler.random(n=n_sim)
    elif ornekleme_yontemi == "qmc":
        sampler = qmc.Sobol(d=boyut, seed=seed, scramble=True)
        m = int(np.ceil(np.log2(max(n_sim, 2))))
        u = sampler.random_base2(m=m)
        return u[:n_sim]
    else:
        return rng.random((n_sim, boyut))


def sok_donustur(u_sutunu, loc, scale, df=None):
    """
    [0,1) araligindaki u degerlerini, istenen dagilima donusturur (inverse-CDF /
    quantile transform yontemiyle - hangi orneklemeden geldigine bakmaksizin ayni
    islem calisir, bu yuzden SRS/LHS/QMC ile uyumludur).
    df=None ise Normal(loc, scale); df bir sayi ise Student-t(df, loc, scale)
    kullanilir (kucuk-orneklem belirsizligini hesaba katan 'Bayesian' duzeltme).
    """
    if df is not None:
        return t_dagilim.ppf(u_sutunu, df=max(df, 1e-6), loc=loc, scale=scale)
    return norm_dagilim.ppf(u_sutunu, loc=loc, scale=scale)


def ulusal_etki_hesapla(tur_idx, gecis_yili_haric=True):
    """
    COK SEVIYELI SIMULASYONUN 1. SEVIYESI (ULUSAL/SISTEMIK):
    Her yil-gecisi (2022->2023, 2023->2024, 2024->2025) icin TUM universitelerin
    o gecisteki farklarinin ORTALAMASINI alir - bu, o yil TUM sisteme ayni anda
    etki eden 'ulusal etki'yi temsil eder (kontenjan/mevzuat degisikligi, sinav
    zorlugu, vb - okula ozgu degil).
    gecis_yili_haric=True ise 2023->2024 (bilinen tek seferlik yapisal kirilma,
    muhtemelen Cambridge->MUR gecisi) bu hesaba KATILMAZ - cunku 2026'da
    tekrarlanmasi beklenmiyor, dahil etmek ulusal varyansi yapay sekilde sisirir.
    Doner: (ortalama_ulusal_etki, ulusal_etki_std, {gecis: deger} sozlugu)
    """
    tum_gecisler = [(2022, 2023), (2023, 2024), (2024, 2025)]
    gecisler = [t for t in tum_gecisler if not (gecis_yili_haric and t == (2023, 2024))]
    ulusal = {}
    for t in gecisler:
        diffs = []
        for uni in UNIVERSITELER:
            yillar_puan = TABAN_PUANLAR.get(uni, {})
            v0 = yillar_puan.get(t[0], (None, None))[tur_idx]
            v1 = yillar_puan.get(t[1], (None, None))[tur_idx]
            if v0 is not None and v1 is not None:
                diffs.append(v1 - v0)
        if diffs:
            ulusal[t] = float(np.mean(diffs))
    degerler = list(ulusal.values())
    if len(degerler) >= 2:
        return float(np.mean(degerler)), float(np.std(degerler, ddof=1)), ulusal
    elif len(degerler) == 1:
        return degerler[0], abs(degerler[0]) * 0.5, ulusal
    else:
        return 0.0, 3.0, ulusal


def idiosinkratik_kalinti_hesapla(uni, tur_idx, ulusal_sozluk, gecis_yili_haric=True):
    """
    COK SEVIYELI SIMULASYONUN 2. SEVIYESI (OKULA OZGU/IDIOSINKRATIK):
    Bu okulun HER yil-gecisindeki gercek farkindan, o gecisin ULUSAL ETKISINI
    cikarir. Kalan 'kalinti' (residual), o okula OZGU sapmayi temsil eder -
    ulusal ortak sokun etkisi zaten ayristirilmis oldugu icin bu kalintilarin
    varyansi ham yil-yil farklardan cok daha kucuk ve gercekci cikar.
    """
    yillar_puan = TABAN_PUANLAR.get(uni, {})
    tum_gecisler = [(2022, 2023), (2023, 2024), (2024, 2025)]
    gecisler = [t for t in tum_gecisler if not (gecis_yili_haric and t == (2023, 2024))]
    kalintilar = []
    for t in gecisler:
        v0 = yillar_puan.get(t[0], (None, None))[tur_idx]
        v1 = yillar_puan.get(t[1], (None, None))[tur_idx]
        if v0 is not None and v1 is not None and t in ulusal_sozluk:
            kalintilar.append((v1 - v0) - ulusal_sozluk[t])
    return kalintilar


def idiosinkratik_volatilite_havuzlanmis(tur_idx, ulusal_sozluk, gecis_yili_haric=True):
    """Tum universitelerin kalintilarindan klasik pooled variance - shrinkage icin taban."""
    pay, payda, hepsi = 0.0, 0, []
    for uni in UNIVERSITELER:
        kalintilar = idiosinkratik_kalinti_hesapla(uni, tur_idx, ulusal_sozluk, gecis_yili_haric)
        hepsi.extend(kalintilar)
        if len(kalintilar) >= 2:
            var_i = float(np.var(kalintilar, ddof=1))
            df_i = len(kalintilar) - 1
            pay += df_i * var_i
            payda += df_i
    if payda == 0:
        return float(np.var(hepsi)) if hepsi else 1.0
    return pay / payda


def cok_seviyeli_simulasyon(uni, tur_idx, n_sim=100000, seed=42, sensitivite=0.75, k=3, gecis_yili_haric=True,
                              ornekleme_yontemi="srs", kucuk_ornek_duzeltmesi=False):
    """
    COK SEVIYELI (HIERARCHICAL / MULTI-LEVEL) MONTE CARLO SIMULASYONU:
      SEVIYE 1 - Ulusal: her simulasyon orneginde BIR KERE, TUM universiteler
        icin ORTAK bir 'ulusal sok' orneklenir
      SEVIYE 2 - Okula ozgu: o okula ait, ulusal etki ayristirildiktan SONRA
        kalan kalintilardan turetilen (shrinkage/pooled) idiosinkratik std ile orneklenir
      TOPLAM SICRAMA = Seviye1 + Seviye2 (+ kontenjan baski etkisi)

    ornekleme_yontemi: 'srs' (Basit Rastgele), 'lhs' (Katmanli/Latin Hypercube),
        'qmc' (Quasi-Monte Carlo/Sobol) - hangisi secilirse secilsin AYNI dagilimdan
        (Normal ya da Student-t) ornek uretir, sadece [0,1) noktalarinin NASIL
        yerlestirildigi degisir.
    kucuk_ornek_duzeltmesi: True ise Normal yerine Student-t dagilimi kullanilir -
        cunku sadece 1-2 veri noktasindan hesaplanan bir ortalama/std'yi KESIN
        biliyormus gibi davranmak (Normal) kucuk orneklemlerde belirsizligi
        OLDUGUNDAN AZ gosterir. Student-t, bu ek belirsizligi otomatik olarak
        hesaba katar (serbestlik deresi ne kadar kucukse kuyruklar o kadar
        kalinlasir) - bu, tam bir MCMC calistirmaya gerek kalmadan (cunku model
        conjugate/kapali-form cozumlu) ayni sonucu veren dogru istatistiksel yol.
    """
    yillar_puan = TABAN_PUANLAR.get(uni, {})
    degerler = [(y, yillar_puan[y][tur_idx]) for y in [2022, 2023, 2024, 2025]
                if yillar_puan.get(y, (None, None))[tur_idx] is not None]
    if len(degerler) < 2:
        return None
    son_yil, son_puan = degerler[-1]

    ulusal_ort, ulusal_std, ulusal_sozluk = ulusal_etki_hesapla(tur_idx, gecis_yili_haric)
    ulusal_df = max(len(ulusal_sozluk) - 1, 1)
    kalintilar = idiosinkratik_kalinti_hesapla(uni, tur_idx, ulusal_sozluk, gecis_yili_haric)
    pooled_idio_var = idiosinkratik_volatilite_havuzlanmis(tur_idx, ulusal_sozluk, gecis_yili_haric)

    if len(kalintilar) >= 1:
        df_i = len(kalintilar) - 1
        if df_i >= 1:
            individual_var = float(np.var(kalintilar, ddof=1))
            shrunk_idio_var = (df_i * individual_var + k * pooled_idio_var) / (df_i + k)
        else:
            shrunk_idio_var = pooled_idio_var
        idio_ortalama = float(np.mean(kalintilar))
    else:
        shrunk_idio_var = pooled_idio_var
        idio_ortalama = 0.0
    idio_std = float(np.sqrt(max(shrunk_idio_var, 0.0)))
    idio_df = max(len(kalintilar) - 1 + k, 1)  # shrinkage'daki k, havuzdan gelen "ek gozlem" gibi df'e katiliyor

    yillar_koltuk = KONTENJAN_TARIHSEL.get(uni, {})
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

    u = taban_ornek_uret(n_sim, seed, boyut=2, ornekleme_yontemi=ornekleme_yontemi)
    df1 = ulusal_df if kucuk_ornek_duzeltmesi else None
    df2 = idio_df if kucuk_ornek_duzeltmesi else None
    seviye1_ulusal_sok = sok_donustur(u[:, 0], loc=ulusal_ort, scale=max(ulusal_std, 0.01), df=df1)
    seviye2_okul_soku = sok_donustur(u[:, 1], loc=idio_ortalama, scale=max(idio_std, 0.01), df=df2)
    sonuclar = son_puan + seviye1_ulusal_sok + seviye2_okul_soku + baski_etkisi
    sonuclar = np.clip(sonuclar, 0.0, 90.0)
    sonuclar_sirali = np.sort(sonuclar)

    p95 = round(float(np.percentile(sonuclar_sirali, 95)), 1)
    return {
        "son_yil": son_yil, "son_puan": son_puan,
        "ulusal_ortalama": round(ulusal_ort, 2), "ulusal_std": round(ulusal_std, 2),
        "idiosinkratik_std": round(idio_std, 2),
        "medyan": round(float(np.median(sonuclar_sirali)), 1),
        "p95_tahmin_araligi": p95,
        "p5_kotumser": round(float(np.percentile(sonuclar_sirali, 5)), 1),
        "guvenilir": p95 < GUVENILIRLIK_ESIGI,
        "n_sim": n_sim,
    }


def monte_carlo_simulasyon(uni, tur_idx, n_sim=100000, seed=42, sensitivite=0.75, k=3, gecis_yili_haric=True):
    """
    SEFFAF Monte Carlo simulasyonu - her adimi acik:
      1) 2022-2025 arasi ardisik yil farklarini toplar (gecis_yili_haric=True ise
         2023->2024 sistemik/tek seferlik sicramasi disarida birakilir - bkz. _farklar_bul)
      2) Ortalama farki hesaplar (ort_fark) - eger cikarma sonrasi hic fark kalmadiysa
         (ort_fark hesaplanamiyorsa), TUM okullarin (ayni haric tutma ile) ortalama
         farkina guvenilir (asagida ele alinir)
      3) VOLATILITE = volatilite_index_hesapla() ile shrinkage/pooled yontemle hesaplanir
      4) n_sim (varsayilan 100.000) adet rastgele "2026 sicramasi" ornekler:
         Normal(ort_fark, volatilite) dagiliminda
      5) Her orneklenen sicramaya, taban_tahmini_hesapla'daki AYNI kontenjan
         baski etkisi eklenir (tutarlilik icin)
      6) Sonuclar 0-90 araliginda sinirlanir (60 soru x 1.5 puan = matematiksel tavan)
      7) SIMULASYON MEDYANI = n_sim sonucu kucukten buyuge SIRALAYIP tam ortadaki
         degeri almak (np.median)
      8) %95 TAHMIN ARALIGI UST SINIRI = sirali sonuclarin 95. yuzdelik dilimi.
         DIKKAT: bu klasik anlamda bir 'confidence interval' DEGIL - CI, tekrarlanan
         orneklemede GERCEK POPULASYON PARAMETRESINI icerecek araligi tanimlar.
         Burada urettigimiz sey bir 'PREDICTION INTERVAL' (tahmin araligi).
    """
    yillar_puan = TABAN_PUANLAR.get(uni, {})
    degerler = [(y, yillar_puan[y][tur_idx]) for y in [2022, 2023, 2024, 2025]
                if yillar_puan.get(y, (None, None))[tur_idx] is not None]
    if len(degerler) < 2:
        return None
    son_yil, son_puan = degerler[-1]

    farklar = _farklar_bul(uni, tur_idx, gecis_yili_haric=gecis_yili_haric)
    if farklar:
        ort_fark = float(np.mean(farklar))
        ort_fark_kaynak = "okulun kendi verisi"
    else:
        # gecis yili haric tutulunca bu okulda hic fark kalmadi (orn. 2 veri
        # noktasi vardi ve tam olarak 2023->2024 araligiydi) - tum okullarin
        # (ayni haric tutmayla) ortalama farkina guveniliyor, acikca belirtiliyor
        tum_farklar = []
        for u in UNIVERSITELER:
            tum_farklar.extend(_farklar_bul(u, tur_idx, gecis_yili_haric=gecis_yili_haric))
        ort_fark = float(np.mean(tum_farklar)) if tum_farklar else 0.0
        ort_fark_kaynak = "TUM okullarin ortalamasi (bu okulda gecerli fark kalmadi)"

    volatilite = volatilite_index_hesapla(uni, tur_idx, k=k, gecis_yili_haric=gecis_yili_haric)

    yillar_koltuk = KONTENJAN_TARIHSEL.get(uni, {})
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

    rng = np.random.default_rng(seed)
    ornek_siçramalar = rng.normal(loc=ort_fark, scale=max(volatilite, 0.01), size=n_sim)
    sonuclar = son_puan + ornek_siçramalar + baski_etkisi
    sonuclar = np.clip(sonuclar, 0.0, 90.0)
    sonuclar_sirali = np.sort(sonuclar)

    p95_duz = round(float(np.percentile(sonuclar_sirali, 95)), 1)
    return {
        "son_yil": son_yil, "son_puan": son_puan,
        "ort_fark": round(ort_fark, 2), "ort_fark_kaynak": ort_fark_kaynak,
        "volatilite": round(volatilite, 2),
        "medyan": round(float(np.median(sonuclar_sirali)), 1),
        "p95_tahmin_araligi": p95_duz,
        "p5_kotumser": round(float(np.percentile(sonuclar_sirali, 5)), 1),
        "guvenilir": p95_duz < GUVENILIRLIK_ESIGI,
        "n_sim": n_sim,
    }


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
def sansa_birakma_analizi(n_soru, elenebilen_sik, dogru_puan=1.5, yanlis_puan=-0.4, toplam_sik=5):
    """
    BINOM DAGILIMI ile 'bilmedigin sorulari sansa birakirsan ne olur' analizi.
    Her soruda 'toplam_sik' secenek var; elenebilen_sik kadarini eleyebiliyorsan,
    kalan (toplam_sik - elenebilen_sik) secenek arasinda rastgele isaretliyorsun ->
    dogru olma olasiligi p = 1 / (toplam_sik - elenebilen_sik).
    n_soru tane BAGIMSIZ boyle soru isaretlersen, dogru sayin X ~ Binom(n_soru, p)
    dagilimini izler. Bu fonksiyon: p, beklenen deger (tek soru ve toplam), ve X'in
    olasi her degeri icin toplam puan katkisi + olasiligini dondurur.
    """
    if elenebilen_sik >= toplam_sik:
        return None
    p = 1.0 / (toplam_sik - elenebilen_sik)
    ev_soru = p * dogru_puan + (1 - p) * yanlis_puan
    ev_toplam = ev_soru * n_soru

    dagilim = []
    olumlu_olasilik = 0.0  # P(toplam katki >= 0)
    for x in range(0, n_soru + 1):
        olasilik = float(binom_dagilim.pmf(x, n_soru, p))
        katki = dogru_puan * x + yanlis_puan * (n_soru - x)
        if katki >= 0:
            olumlu_olasilik += olasilik
        dagilim.append({"Dogru Sayisi (X)": x, "Puan Katkisi": round(katki, 2), "Olasilik": round(olasilik, 4)})

    return {
        "p": p, "ev_soru": round(ev_soru, 3), "ev_toplam": round(ev_toplam, 2),
        "olumlu_olasilik": round(olumlu_olasilik, 3), "dagilim": dagilim,
    }


def gerekli_dogru_hesapla(hedef_puan, bos_sayisi, toplam_soru=60, dogru_puan=1.5, yanlis_puan=-0.4):
    """
    Sabit bir 'bos sayisi' icin, hedef puana ULASMAK (ya da gecmek) icin gereken
    EN AZ dogru sayisini hesaplar. Kalan sorular otomatik olarak yanlis kabul edilir
    (en kotu ihtimal / garanti esik).
    Formul: 1.5*D - 0.4*Y >= hedef, D+Y = toplam_soru - bos_sayisi
         => D >= (hedef + 0.4*(toplam_soru - bos_sayisi)) / (1.5 - (-0.4))
    Doner: (gereken_en_az_dogru, buna_karsilik_yanlis, gercek_elde_edilen_puan) ya da
    None (bu kadar bos ile hedefe ulasmak MATEMATIKSEL OLARAK imkansizsa).
    """
    kalan = toplam_soru - bos_sayisi
    if kalan < 0:
        return None
    maks_mumkun_puan = dogru_puan * kalan
    if maks_mumkun_puan < hedef_puan - 1e-9:
        return None
    payda = dogru_puan - yanlis_puan
    pay = hedef_puan - yanlis_puan * kalan
    d_ham = pay / payda
    d_min = max(0, math.ceil(d_ham - 1e-9))
    if d_min > kalan:
        return None
    yanlis = kalan - d_min
    gercek_puan = round(dogru_puan * d_min + yanlis_puan * yanlis, 2)
    return d_min, yanlis, gercek_puan


def yanlis_sweep_tablosu(hedef_puan, toplam_soru=60, dogru_puan=1.5, yanlis_puan=-0.4):
    """Bos=0 sabitken, her mumkun yanlis sayisi icin gereken en az dogruyu tablolar."""
    satirlar = []
    for y in range(0, toplam_soru + 1):
        payda = dogru_puan - yanlis_puan
        pay = hedef_puan - yanlis_puan * y
        d_ham = pay / payda
        d_min = max(0, math.ceil(d_ham - 1e-9))
        if d_min + y > toplam_soru:
            break
        bos = toplam_soru - d_min - y
        gercek_puan = round(dogru_puan * d_min + yanlis_puan * y, 2)
        satirlar.append({"Yanlis": y, "Gereken En Az Dogru": d_min, "Bos": bos, "Elde Edilen Puan": gercek_puan})
    return satirlar


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

sekme1, sekme2, sekme3, sekme4, sekme5, sekme6, sekme7, sekme8, sekme9, sekme10 = st.tabs([
    "Dogrulanmis Temel Veri", "Senaryo Simulatoru", "Ozel Taban Puan Ekleyici",
    "En Kotu Senaryo (Worst Case)", "Merkezi Tahmin (Ana Senaryo)", "Yerlesim Tahmini (Tercih Listem)",
    "Terimler & Metodoloji", "Dis Kaynak Karsilastirma", "Seffaf Monte Carlo Simulasyonu",
    "Puan Hedefi Hesaplayici"
])

# ----------------------------------------------------------------------
# SEKME 1 - Dogrulanmis temel veri, goz atma
# ----------------------------------------------------------------------
with sekme1:
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Burada gecmis yillarda GERCEKTEN olan sinav "
        "sonuclari var - hicbir sey tahmin degil, hepsi olmus bitmis, gercek. Bir fotograf "
        "albumu gibi dusun: gecmiste ne olduysa onu gosteriyor, gelecegi tahmin etmiyor.\n\n"
        "**Nasil kullanilir?** Hicbir seye dokunmana gerek yok - sadece asagi kaydir ve oku. "
        "Butun diger sekmelerdeki tahminler, buradaki gercek sayilardan yola cikiyor."
    )
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
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Burada 'ya boyle olursa?' oyunu oynuyoruz. Mesela: "
        "'ya sinava daha cok kisi girerse?' ya da 'ya kontenjan artarsa?' Sen bir ihtimal "
        "hayal ediyorsun, program da 'o zaman taban puan su olur' diye tahmin ediyor.\n\n"
        "**Nasil kullanilir?** Asagidaki kaydirma cubuklarini (slider) saga sola cek - her "
        "cektiginde sayilar degisir. Kendi hayalindeki senaryoyu kur, sonucu asagidaki "
        "tabloda gor."
    )
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
        tahmin_eu = None if eu_2025 is None else max(0.0, min(90.0, round(eu_2025 + tahmini_degisim_eu - zorluk_degisimi, 1)))
        tahmin_non = None if non_2025 is None else max(0.0, min(90.0, round(non_2025 + tahmini_degisim_noneu - zorluk_degisimi, 1)))
        tahmin_satirlari.append({"Universite": uni, "2025_EU": eu_2025, "Tahmini_2026_EU": tahmin_eu,
                                  "2025_NonEU": non_2025, "Tahmini_2026_NonEU": tahmin_non})
    st.dataframe(pd.DataFrame(tahmin_satirlari), use_container_width=True)

# ----------------------------------------------------------------------
# SEKME 3 - Universite ve yil bazinda ozel senaryo ekleme
# ----------------------------------------------------------------------
with sekme3:
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Burada TAMAMEN kendi hayalindeki bir durumu "
        "deneyebilirsin - mesela 'ya Bari'nin puani 2 puan duserse?' Bunu SEN soyluyorsun, "
        "program gercek veriyle karsilastirip ne olacagini gosteriyor.\n\n"
        "**Nasil kullanilir?** 1) Bir okul sec. 2) EU mu Non-EU mu oldugunu sec. 3) Hangi "
        "yili baz almak istedigini sec. 4) Puani kac degistirmek istedigini yaz. Istersen "
        "'gunluge ekle' diyerek bu senaryoyu kaydedebilirsin, sonra tekrar bakarsin."
    )
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
    st.info(
        "🧒 **Bu sekme ne ise yarar?** 'En kotu ihtimalde ne olur?' diye bakiyoruz - senin "
        "hazirlikli olman icin. Sokaga cikarken 'yagmur yagmayabilir ama semsiyemi yine de "
        "alayim' demek gibi dusun. Gecmiste bir okulun puani en cok ne kadar birden zipladiysa, "
        "'yine oyle olursa' diye varsayiyoruz.\n\n"
        "**Nasil kullanilir?** Asagi kaydir, tabloyu oku - en yuksek 'En Kotu Senaryo 2026' "
        "sayisi olan okullar en riskli okullar. En altta '5 en riskli' listesi hazir seklinde "
        "de var."
    )
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
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Burada 'en kotu' degil, 'en OLASI' tahmini "
        "gosteriyoruz - ne cok iyimser ne cok kotumser, ortasini buluyoruz. Bir siniftaki "
        "cocuklarin boyunun ortalamasini bulmak gibi dusun - en uzun cocugu degil, "
        "ortalamayi aliyoruz.\n\n"
        "**Nasil kullanilir?** Iki farkli yontemi (Trend ve Momentum) karsilastirip aradaki "
        "kaydirma cubugunu oynatarak ikisine ne kadar agirlik vermek istedigini secebilirsin. "
        "Asagida her okul icin tahmini tabloyu goreceksin."
    )
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
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Bu sekme SENIN icin! Puanini ve istedigin okullari "
        "yaziyorsun, program da 'muhtemelen buraya girersin' diye tahmin ediyor - tipki "
        "bir oyunda hangi seviyeyi gecebilecegini hesaplamak gibi.\n\n"
        "**Nasil kullanilir?** 1) Puanini yaz. 2) EU mu Non-EU mu oldugunu sec. 3) Istedigin "
        "okullari ONCELIK SIRASINA gore tikla (once en cok istedigini). 4) Asagidaki tabloda "
        "her okul icin 'GECER' mi 'ACIK VAR' mi yaziyor gorursun."
    )
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

# ----------------------------------------------------------------------
# SEKME 7 - TERIMLER & METODOLOJI: her terimin ve her hesabin acik aciklamasi
# ----------------------------------------------------------------------
with sekme7:
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Diger sekmelerde zor kelimeler gordugunde ('volatilite', "
        "'Base Case' gibi) buraya gel - hepsini en basit sekilde, ana dilinde anlatiyoruz. "
        "Hicbir hesap yapmiyoruz, sadece kelimeleri aciyoruz.\n\n"
        "**Nasil kullanilir?** Merak ettigin kelimeyi bul, oku. Sirayla okumana gerek yok."
    )
    st.subheader("Terimler Sozlugu ve Metodoloji")
    st.caption("Bu sekme hicbir hesap yapmaz - sadece diger sekmelerde gordugun terimleri ve formulleri acar.")

    st.markdown("### Puanlama Sistemi (IMAT sinavi)")
    st.markdown(
        "🧒 Sinavda 60 soru var. Her DOGRU cevap sana 1.5 puan verir, her YANLIS cevap "
        "senden 0.4 puan alir, BOS birakirsan hicbir sey olmaz. Yani en fazla alabilecegin "
        "puan, 60 sorunun HEPSINI dogru yaparsan olur: 60 x 1.5 = **90 puan**. Bu, asilamayan "
        "bir tavan - tipki bir odanin tavanindan daha yukari zipliyamamana benzer. Bu yuzden "
        "simulatordeki HICBIR tahmin 90'i gecemez, biz de artik hicbir yerde gecirmiyoruz."
    )

    st.divider()
    st.markdown("### Base Case / Merkezi Tahmin / En Kotu Senaryo farki")
    st.markdown(
        "🧒 Ucunu de yaparken ayni oyuncagi kullaniyoruz, sadece 'gecmiste ne olmustu' diye "
        "sordugumuzda farkli yillara bakiyoruz - tipki bir arkadasina 'geciken otobus ne "
        "kadar gecikir' diye sorarken, ya SADECE dun gecikmeyi (Base Case), ya TUM haftanin "
        "ortalamasini (Merkezi), ya da hic gormedigin en kotu gecikmeyi (En Kotu) sormak "
        "gibi:\n\n"
        "- **Base Case**: 'geçen sene ne kadar degistiyse, bu sene de o kadar degisir' diyoruz "
        "- sadece en SON iki yila bakiyoruz, en 'taze' tahmin.\n"
        "- **Merkezi Tahmin**: 'ortalama ne kadar degisti' diyoruz - butun yillari topluyoruz, "
        "ortasini buluyoruz. Tek bir garip yildan (mesela cok tuhaf bir yildan) daha az "
        "etkilenir, en 'dengeli' tahmin.\n"
        "- **En Kotu Senaryo**: 'en cok ne zaman zipladiysa, YINE oyle zipladigini' varsayiyoruz "
        "- korkuluk gibi, en kotu ihtimale hazirlik.\n\n"
        "Ucune de bir de kucuk bir 'itis' ekliyoruz: eger 2026'da o okulun kontenjani (yer "
        "sayisi) fazla artmadiysa, taban puani biraz daha yukari itiyoruz - cunku daha az yer "
        "= daha cok yarisma = daha yuksek puan gerekir, tipki oyuncakcida az oyuncak kalinca "
        "herkesin ona daha cok kosmasi gibi."
    )

    st.divider()
    st.markdown("### Yerlesim Tahmini sekmesindeki kolonlar")
    st.markdown(
        "🧒 - **Son Yil / Son Puan**: o okul icin bildigimiz EN YENI gercek sonuc, hangi yildan\n"
        "- **Base Case 2026 / Merkezi 2026 / En Kotu 2026**: yukarida anlatilan uc farkli tahmin yontemi\n"
        "- **Fark (Base Case)**: senin puanin EKSI Base Case tahmini. Artiysa (+) yeterli demektir, "
        "eksiyse (-) o kadar puana daha ihtiyacin var demektir\n"
        "- **Durum**: Fark sifir ya da daha buyukse 'GECER' yaziyor, degilse 'ACIK VAR' yaziyor - "
        "'acik' senin eksigin demek"
    )

    st.divider()
    st.markdown("### Diger terimler")
    st.markdown(
        "🧒 - **EU / Yerli Havuzu vs Non-EU Havuzu**: Italya'da iki ayri kuyruk var - biri AB "
        "vatandaslari icin, biri AB disindan/yurt disindan gelenler icin. Ayni sinavi "
        "veriyorsunuz ama kuyruklar ayri, yani rekabet de ayri.\n"
        "- **Rekabet Endeksi**: kac kisi yariyor / kac yer var. Bu sayi buyudukce (cok kisi, az "
        "yer) taban puanin da yukselmesi beklenir - tipki bir oyuncak dukkaninda az oyuncak "
        "kalinca fiyatinin artmasi gibi.\n"
        "- **Scorrimento**: ilk yerlesimden sonra bazi kisiler vazgecince, bosalan yerler "
        "SIRADAKI kisilere otomatik verilir - bu birkac ay surer, bir kuyrukta yavas yavas "
        "one gitmek gibi dusun.\n"
        "- **Bosluk yili (⚠️)**: o okul icin bazi yillarin verisi eksik/kayip - yani daha az "
        "bilgiyle tahmin yaptik, biraz daha temkinli bak.\n"
        "- **Tahmin Araligi vs Guven Araligi ve Cok Seviyeli Simulasyon**: bunlar 'Seffaf Monte "
        "Carlo Simulasyonu' sekmesinde en basit dille anlatiliyor - oraya bak."
    )

# ----------------------------------------------------------------------
# SEKME 8 - DIS KAYNAK KARSILASTIRMA: yuklenen PDF raporlarindaki veriyle kiyas
# ----------------------------------------------------------------------
with sekme8:
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Sen bana iki PDF gonderdin - baska birinin yaptigi "
        "tahminler. Biz de bizim tahminlerimizi onlarinkiyle yan yana koyduk, 'ikisi birbirine "
        "benziyor mu?' diye bakiyoruz. Iki farkli kisinin ayni bulmacayi cozup cevaplarini "
        "karsilastirmasi gibi dusun.\n\n"
        "**Nasil kullanilir?** Asagi kaydir, iki tabloya bak. Sayilar birbirine yakinsa, ikisi "
        "de muhtemelen dogruya yakin. Cok farkliysa, en azindan biri (ya da ikisi de) "
        "yanilmis olabilir."
    )
    st.subheader("Dis Kaynak Tahminleriyle Karsilastirma")
    st.warning(
        "Bu sekmedeki veri BU PROJENIN TEMEL VERISI DEGILDIR - ucretli/harici bir 'tahmin' urununden "
        "(IMAT Hero + bir baska ozel rapor) alinmis referans amacli veridir. Metodolojisi seffaf degil: her "
        "universitede 'Simulasyon Medyani' aynen '2025 Final Cut-off' degerine esit cikiyor - yani model gercekte "
        "2025'in etrafina bir 'volatilite' katsayisini standart sapma gibi kullanip rastgele gurultu ekliyor, "
        "gercek bir trend/buyume ongorusu icermiyor. Birincil karar kaynagi olarak degil, ek bir referans noktasi "
        "olarak kullan."
    )

    dis_kaynak_df = pd.DataFrame([
        ["La Sapienza",        61.8, 65.8, 69.5, 80.3, 3.3, 7.6],
        ["Bologna",            60.5, 70.3, 69.5, 79.2, 4.3, 4.2],
        ["Milano Statale",     66.7, 72.9, 74.4, 80.6, 1.1, 2.8],
        ["Milano Bicocca",     64.5, 65.1, 72.2, 79.4, 0.3, 7.5],
        ["Padova",             59.7, 65.4, 67.4, 77.6, 3.5, 6.2],
        ["Pavia",              60.4, 64.8, 68.1, 77.3, 1.3, 6.4],
        ["Torino",             59.5, 67.1, 67.2, 75.2, 0.0, 3.7],
        ["Tor Vergata",        57.9, 69.1, 65.6, 85.0, 1.6, 8.5],
        ["Napoli Federico II", 59.8, 63.1, 67.5, 73.3, 1.6, 5.0],
        ["Parma",              58.0, 67.6, 65.7, 83.6, 0.4, 8.5],
        ["Verona",             57.9, 66.2, 65.6, 75.1, 0.6, 4.2],
        ["Catania",            55.0, 61.6, 62.7, 70.8, 0.3, 4.4],
        ["Luigi Vanvitelli",   56.0, 56.9, 63.7, 69.3, 1.3, 6.3],
        ["Messina",            56.0, 58.2, 63.7, 65.9, 0.9, 3.2],
        ["Cagliari",           55.9, 54.2, 63.6, 76.5, 0.9, 12.3],
        ["Bari",               55.8, 49.3, 63.5, 78.4, 0.2, 16.5],
    ], columns=["Universite", "DisKaynak_2025_Cutoff_EU", "DisKaynak_2025_Cutoff_NonEU",
                "DisKaynak_SafeTarget95_EU", "DisKaynak_SafeTarget95_NonEU",
                "DisKaynak_Volatility_EU", "DisKaynak_Volatility_NonEU"])

    st.markdown("### Dis Kaynak Verisi (oldugu gibi)")
    st.dataframe(dis_kaynak_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Kendi Modelinle Karsilastirma")
    st.caption(
        "Asagida dis kaynagin '%95 Safe Target' degeri (guvenlik payli tavan tahmini) ile senin Base Case / "
        "Merkezi / En Kotu Senaryo tahminlerin yan yana. Dis kaynagin 'Safe Target'i tanim geregi senin En Kotu "
        "Senaryo'na en yakin kavram - ikisini birlikte oku."
    )

    karsilastirma_turu = st.radio("Kontenjan Turu", ["EU", "NonEU"], horizontal=True, key="s8_tur")
    tur_idx_s8 = 0 if karsilastirma_turu == "EU" else 1

    karsilastirma_satirlari = []
    for _, satir in dis_kaynak_df.iterrows():
        uni = satir["Universite"]
        if uni not in UNIVERSITELER:
            continue
        _, _, _, _, tahmin_bc = taban_tahmini_hesapla(uni, tur_idx_s8, yontem="son_fark", sensitivite=0.75)
        _, _, _, _, tahmin_merkezi = taban_tahmini_hesapla(uni, tur_idx_s8, yontem="ortalama", sensitivite=0.75)
        _, _, _, _, tahmin_worst = taban_tahmini_hesapla(uni, tur_idx_s8, yontem="max", sensitivite=1.5)

        dis_safe_target = satir[f"DisKaynak_SafeTarget95_{karsilastirma_turu}"]

        karsilastirma_satirlari.append({
            "Universite": uni,
            "Base Case (bizim)": tahmin_bc,
            "Merkezi (bizim)": tahmin_merkezi,
            "En Kotu (bizim)": tahmin_worst,
            "Dis Kaynak Safe Target (%95)": dis_safe_target,
            "Fark (Dis - Bizim En Kotu)": None if tahmin_worst is None else round(dis_safe_target - tahmin_worst, 1),
        })

    st.dataframe(pd.DataFrame(karsilastirma_satirlari), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# SEKME 9 - SEFFAF MONTE CARLO SIMULASYONU: gercek simulasyon, formul acik
# ----------------------------------------------------------------------
with sekme9:
    st.info(
        "🧒 **Bu sekme ne ise yarar?** Burada bir ZAR OYUNU oynuyoruz - ama gercek bir zar "
        "degil, bilgisayarin icinde 100 BIN kere sanal bir zar atiyoruz. Her atista '2026'da "
        "bu okulun puani ne olabilir' diye bir tahmin cikiyor. 100 bin tahmini kucukten "
        "buyuge diziyoruz, ortasina bakiyoruz, en yuksek uctekilere bakiyoruz - boylece hem "
        "'en olasi' hem 'en kotu ihtimalde ne olur' sorularina cevap buluyoruz.\n\n"
        "**Nasil kullanilir?** Asagida 2 secenek var (Tek Seviyeli / Cok Seviyeli) - hangisini "
        "secersen sec, tabloyu okumak yeterli. Merak edersen alttaki acilir kutulari (▸) "
        "tikla, her sey en basit dilde anlatiliyor."
    )
    st.subheader("Seffaf Monte Carlo Simulasyonu")
    st.caption(
        "Bu sekme dis kaynagin yaptigini iddia ettigi seyi GERCEKTEN yapiyor - ve her adimi "
        "acikca gosteriyor."
    )

    st.error(
        "**Onemli duzeltme**: 🧒 daha once, zar oyunumuz bazen 'en yuksek mumkun puan olan "
        "90'i al, o zaman guvendesin' gibi sacma bir sey soyluyordu. Bu, '90 almazsan girmeni "
        "GARANTI EDEMEYIZ' demenin sacma bir yoluydu - cunku 90 zaten hic kimsenin alamayacagi "
        "kadar yuksek (60 sorunun HEPSI dogru olmali). Bunu duzelttik: artik boyle durumlarda "
        "sayi yazmiyoruz, direkt **'GUVENILMEZ'** yaziyoruz - yani 'biz de bilmiyoruz, veri "
        "yetersiz' diyoruz, uydurmuyoruz."
    )

    with st.expander("🧒 'Tahmin Araligi' ile 'Guven Araligi' ayni sey mi?"):
        st.markdown(
            "Hayir, ikisi FARKLI sorulara cevap verir - kucuk bir ornekle anlatalim:\n\n"
            "Diyelim sinifin ortalama boyunu merak ediyorsun. **Guven araligi**, 'sinifin "
            "GERCEK ortalama boyu 140-145 cm arasindadir' gibi bir cevap verir - yani "
            "GRUBUN ozelligi hakkinda.\n\n"
            "Ama biz farkli bir soru soruyoruz: 'YARIN sinifa YENI gelecek TEK bir cocugun "
            "boyu ne olur?' Bu soru daha zor, cunku sadece grubun ortalamasini degil, o "
            "tek cocugun kendi farklarini da hesaba katmali. Bunun cevabina **tahmin araligi** "
            "denir, ve her zaman guven araligindan biraz daha genis cikar - cunku iki turlu "
            "belirsizligi birden tasir. Biz burada 'gelecek yilin TEK bir taban puani ne "
            "olur' diye sordugumuz icin, dogru arac tahmin araligidir."
        )

    with st.expander("🧒 Volatilite (oynaklik) ne demek, nasil hesapladik?"):
        st.markdown(
            "**Volatilite**, bir seyin yildan yila NE KADAR ZIPLADIGINI olcer. Bir topu "
            "dusun: bazi toplar hafif ziplar (dusuk volatilite), bazilari cok sert ziplar "
            "(yuksek volatilite).\n\n"
            "Sorun su: bir okulun sadece 2-3 yillik verisi var, bu COK az. Sadece 2-3 "
            "ziplamaya bakip 'bu topun ziplama gucu tam olarak budur' demek riskli - "
            "sansa da bagli olabilir. Bu yuzden akilli bir numara yapiyoruz: sadece o "
            "okulun kendi ziplamalarina degil, TUM okullarin ziplamalarina birden bakiyoruz, "
            "ve okulun kendi verisi azsa, 'digerleri nasil ziplıyorsa o da oyle ziplar' "
            "diye varsayiyoruz biraz. Verisi ne kadar azsa, digerlerine o kadar cok "
            "guveniyoruz - buna istatistikte 'havuzlama' denir."
        )

    with st.expander("🧒 Tek Seviyeli ile Cok Seviyeli (Multi-Level) arasindaki fark ne?", expanded=True):
        st.markdown(
            "Bunu bir sinif ornegiyle anlatalim. Diyelim butun sinif ayni gun grip oldu - "
            "HERKESIN notu dustu. Ama Ahmet'in AYRICA kendi sinav korkusu var, o da HER "
            "sinavda biraz daha dusuk not aliyor.\n\n"
            "**Tek Seviyeli** yontem, Ahmet'in notundaki degisimi TEK bir sebebe bagliyor - "
            "grip mi, korku mu, ayirt etmiyor, hepsini birlikte 'Ahmet'in oynakligi' sayiyor. "
            "Bu yanlis olur, cunku grip HERKESTE var, sadece Ahmet'e ozel degil.\n\n"
            "**Cok Seviyeli** yontem daha akillica: once 'butun sinifi ayni anda etkileyen "
            "sey ne kadardi' diye bakar (grip - SEVIYE 1, ULUSAL), sonra bunu cikarip "
            "'Ahmet'e OZEL ne kadar kaldi' diye bakar (korku - SEVIYE 2, OKULA OZGU). "
            "Iki ayri zar atiyoruz - biri 'bu yil butun okullari birden etkileyen sansizlik' "
            "icin, biri 'sadece bu okula ozel sansizlik' icin - sonra ikisini topluyoruz. "
            "Boylece bir okulun 'gercek' oynakligini, herkesi etkileyen genel gurultudan "
            "ayirabiliyoruz - ki bu cok daha dogru bir cevap verir."
        )

    with st.expander("🧒 '2023->2024'u hesaptan cikardik' derken ne demek istiyoruz?"):
        st.markdown(
            "2023'ten 2024'e gecerken HER okulun puani birden cok fazla yukseldi (~21-22 "
            "puan!) - bu, sinifin toplu grip olmasi gibi, TEK SEFERLIK, ozel bir sebepten "
            "oldu (muhtemelen sinavin yapisi/kurumu degisti). Bunun 2026'da TEKRAR olmasini "
            "beklemiyoruz. Eger bunu hesaba normal bir 'ziplamaymis' gibi katarsak, "
            "makine 'her yil boyle bir sey olabilir' sanip cok abartili tahminler uretir. "
            "Bu yuzden bu ozel yili, 'bu ola olamayan bir sey, tekrar olmaz' diyerek "
            "hesaptan cikariyoruz - istersen asagidaki kutucuktan bunu tekrar acabilirsin."
        )

    with st.expander("🧒 'k' (havuzlama gucu) ne ise yariyor?"):
        st.markdown(
            "Bu bir DUGME - 'okulun kendi verisine mi, yoksa digerlerinin ortalamasina mi "
            "daha cok guveneyim' diye ayarliyor. k KUCUKSE ('1' gibi), okulun kendi verisine "
            "daha cok guveniriz - az veri olsa bile. k BUYUKSE ('10' gibi), 'okulun kendi "
            "verisi cok az, digerlerine bakayim' deriz. Ortasi (3) iyi bir denge - ne cok "
            "kendine guveniyoruz ne cok digerlerine."
        )

    with st.expander("🧒 'GUVENILMEZ' yazan sonuclar ne demek?"):
        st.markdown(
            "Bazi okullarda veri o kadar az/oynak ki, zar oyunumuz cok genis bir sonuc "
            "araligi cikariyor - o kadar genis ki, ust siniri neredeyse imkansiz olan 90'a "
            "(mumkun en yuksek puan) dayaniyor. Boyle bir durumda 'tahminimiz 90' demek "
            "yalan soylemek gibi olur - gercekte 'bilmiyoruz' demek gerekir. Bu yuzden "
            "boyle sonuclari sayi olarak degil, direkt **GUVENILMEZ** yazarak gosteriyoruz - "
            "durustluk, uydurma bir kesinlik gostermekten daha onemli."
        )

    with st.expander("🧒 Orneklem yontemleri: Basit Rastgele / Katmanli / Quasi-Monte Carlo"):
        st.markdown(
            "Zar oyununu OYNAMA SEKLIMIZ 3 farkli olabilir - hepsi ayni ortalama sonuca "
            "gider ama biri digerinden daha 'duzenli':\n\n"
            "- **Basit Rastgele (SRS)**: tam bir zar gibi, her atis tamamen bagimsiz ve "
            "sanslidir. Bazen sans eseri bir bolgeye cok fazla, bir bolgeye az dusebilir.\n"
            "- **Katmanli (Latin Hypercube)**: piyango biletlerini 100 bin esit gruba "
            "ayirip HER gruptan tam olarak bir bilet cekmek gibi dusun - hicbir bolge "
            "'unutulmus' olmaz, daha DUZGUN bir kapsama saglar.\n"
            "- **Quasi-Monte Carlo (Sobol)**: rastgele bile degil - noktalari matematiksel "
            "olarak birbirinden en uzak, en es dagilmis sekilde ONCEDEN yerlestiriyoruz. "
            "En 'temiz' kapsama budur.\n\n"
            "Bu ucunun de SONUCU (medyan, %95 vs.) neredeyse ayni cikar - cunku zaten "
            "100.000 ornek yeterince fazla. Farki asil kucuk orneklem sayilarinda (1000 "
            "gibi) gorursun - Katmanli/QMC daha az orneklede bile duzgun sonuc verir."
        )

    with st.expander("🧒 'Kucuk-Ornek Duzeltmesi' (Student-t / Bayesian) ne demek?"):
        st.markdown(
            "Bize sordugun listede **MCMC** de vardi - onu neden kullanmadigimizi burada "
            "acikliyoruz. MCMC, cok karisik/analitik cozumu olmayan problemlerde 'dolayli "
            "yoldan' ornek uretmek icin kullanilir. Ama bizim problemimizin ZATEN kapali "
            "(analitik) bir cozumu var - bu yuzden MCMC'ye HIC GEREK YOK, dogrudan daha "
            "basit ve KESIN bir yontem kullanabiliriz: **Student-t dagilimi**.\n\n"
            "Sebebi soyle: sadece 1-2 yillik farktan bir ortalama/std hesaplarken, bunu "
            "'KESIN dogru' gibi kullanmak (Normal dagilim) YALAN bir guven verir - cunku "
            "2 sayidan hesaplanan bir ortalama, gercekte ne kadar dogru oldugunu da "
            "bilmiyoruz! Student-t dagilimi bu EK belirsizligi otomatik ekler - ne kadar "
            "AZ veri varsa (dusuk 'serbestlik derecesi'), kuyruklari o kadar KALINLASIR, "
            "yani 'aslinda hicbir sey bilmiyoruz' der gibi daha genis bir aralik verir.\n\n"
            "**Bunu actiginda tahmin araliklari daha da genisleyecek, belki daha cok okul "
            "GUVENILMEZ cikacak - bu uzucu ama DURUST bir sonuc: gercekten bu kadar az "
            "veriyle daha fazla kesinlik iddia edemeyiz.**"
        )

    st.markdown("#### Bize sordugun 5 yontemden hangilerini kullandik?")
    st.markdown(
        "✅ **Basit Rastgele** - hep kullandik (varsayilan)\n\n"
        "✅ **Katmanli (Stratified/LHS)** - simdi eklendi, secenek olarak asagida\n\n"
        "✅ **Quasi-Monte Carlo** - simdi eklendi, secenek olarak asagida\n\n"
        "⚠️ **Onem Orneklemesi (Importance Sampling)**: KULLANMADIK - bu yontem, cok NADIR "
        "gorulen olaylarin olasiligini hesaplarken ise yarar (orn. 'milyonda bir sans "
        "olan bir sey'). Biz zaten butun dagilimin %5-%50-%95'ini hesapliyoruz, nadir bir "
        "olay pesinde degiliz - bu yuzden burada bir faydasi olmazdi.\n\n"
        "⚠️ **MCMC**: KULLANMADIK, yukarida acikladigimiz gibi problemimizin zaten kapali "
        "(analitik) bir cozumu var (Student-t) - MCMC'nin bize saglayacagi ekstra bir sey "
        "yok, sadece gereksiz karmasiklik katardi."
    )

    gecis_yili_secim = st.checkbox(
        "2023->2024 sistemik sicramasini hesaptan cikar (onerilir - Tek Seviyeli icin gerekli, "
        "Cok Seviyeli icin ek guvence)", value=True, key="s9_gecis"
    )
    yontem_s9 = st.radio("Simulasyon Yontemi", ["Tek Seviyeli", "Cok Seviyeli (Hierarchical)"],
                          horizontal=True, key="s9_yontem")
    ornekleme_s9 = st.selectbox(
        "Orneklem Yontemi (sadece Cok Seviyeli'de aktif)",
        ["srs", "lhs", "qmc"],
        format_func=lambda x: {"srs": "Basit Rastgele (SRS)", "lhs": "Katmanli (Latin Hypercube)",
                                "qmc": "Quasi-Monte Carlo (Sobol)"}[x],
    )
    kucuk_ornek_s9 = st.checkbox(
        "Kucuk-Ornek Duzeltmesi (Student-t / Bayesian) kullan - daha DURUST ama daha GENIS araliklar "
        "(sadece Cok Seviyeli'de aktif)",
        value=False, key="s9_kucuk_ornek"
    )
    n_sim_secim = st.select_slider("Orneklem sayisi (n_sim)", options=[1000, 10000, 100000, 500000], value=100000)
    k_secim = st.slider("Havuzlama gucu (k) - yuksek = daha cok havuza guven, dusuk = daha cok okulun kendi verisine guven",
                          1, 10, 3)
    tur_s9 = st.radio("Kontenjan Turu", ["EU", "NonEU"], horizontal=True, key="s9_tur")
    tur_idx_s9 = 0 if tur_s9 == "EU" else 1

    satirlar_s9 = []
    for uni in UNIVERSITELER:
        if yontem_s9 == "Tek Seviyeli":
            sonuc = monte_carlo_simulasyon(uni, tur_idx_s9, n_sim=n_sim_secim, sensitivite=0.75, k=k_secim,
                                             gecis_yili_haric=gecis_yili_secim)
            if sonuc is None:
                continue
            satirlar_s9.append({
                "Universite": uni + (" ⚠️" if "TUM okullarin" in sonuc["ort_fark_kaynak"] else ""),
                "Son Puan": sonuc["son_puan"],
                "Volatilite": sonuc["volatilite"],
                "Simulasyon Medyani": sonuc["medyan"],
                "%95 Tahmin Araligi": sonuc["p95_tahmin_araligi"] if sonuc["guvenilir"] else "GUVENILMEZ",
                "%5 Kotumser Sinir": sonuc["p5_kotumser"],
            })
        else:
            sonuc = cok_seviyeli_simulasyon(uni, tur_idx_s9, n_sim=n_sim_secim, sensitivite=0.75, k=k_secim,
                                              gecis_yili_haric=gecis_yili_secim,
                                              ornekleme_yontemi=ornekleme_s9,
                                              kucuk_ornek_duzeltmesi=kucuk_ornek_s9)
            if sonuc is None:
                continue
            satirlar_s9.append({
                "Universite": uni,
                "Son Puan": sonuc["son_puan"],
                "Ulusal Etki (Seviye 1)": sonuc["ulusal_ortalama"],
                "Ulusal Std": sonuc["ulusal_std"],
                "Okula Ozgu Std (Seviye 2)": sonuc["idiosinkratik_std"],
                "Simulasyon Medyani": sonuc["medyan"],
                "%95 Tahmin Araligi": sonuc["p95_tahmin_araligi"] if sonuc["guvenilir"] else "GUVENILMEZ",
                "%5 Kotumser Sinir": sonuc["p5_kotumser"],
            })

    df_s9 = pd.DataFrame(satirlar_s9)
    st.dataframe(df_s9, use_container_width=True, hide_index=True)

    guvenilmez_sayisi = (df_s9["%95 Tahmin Araligi"] == "GUVENILMEZ").sum()
    if guvenilmez_sayisi > 0:
        st.warning(
            f"{guvenilmez_sayisi} okul icin %95 tahmin araligi GUVENILMEZ olarak isaretlendi "
            f"(esik: {GUVENILIRLIK_ESIGI} puan) - bu okullarin verisi cok az/kararsiz, model "
            "anlamli bir ust sinir uretemiyor. Sayiyi zorla goze carpmiyoruz."
        )
    st.info(
        f"n_sim={n_sim_secim:,} ornek uretildi, kucukten buyuge siralandi, tam ortadaki deger "
        "'Simulasyon Medyani' olarak alindi."
    )

    st.divider()
    st.markdown("### Dis Kaynakla Yan Yana")
    st.caption("Bizim seffaf %95 tahmin araligimiz ile dis kaynagin 'Safe Target'i.")

    dis_kaynak_karsilastirma = pd.DataFrame([
        ["La Sapienza", 69.5, 80.3], ["Bologna", 69.5, 79.2], ["Milano Statale", 74.4, 80.6],
        ["Milano Bicocca", 72.2, 79.4], ["Padova", 67.4, 77.6], ["Pavia", 68.1, 77.3],
        ["Torino", 67.2, 75.2], ["Tor Vergata", 65.6, 85.0], ["Napoli Federico II", 67.5, 73.3],
        ["Parma", 65.7, 83.6], ["Verona", 65.6, 75.1], ["Catania", 62.7, 70.8],
        ["Luigi Vanvitelli", 63.7, 69.3], ["Messina", 63.7, 65.9], ["Cagliari", 63.6, 76.5],
        ["Bari", 63.5, 78.4],
    ], columns=["Universite", "DisKaynak_SafeTarget_EU", "DisKaynak_SafeTarget_NonEU"])

    karsilastirma_s9 = dis_kaynak_karsilastirma.merge(
        df_s9[["Universite", "%95 Tahmin Araligi"]].assign(
            Universite=df_s9["Universite"].str.replace(" ⚠️", "", regex=False)
        ),
        on="Universite", how="left"
    )
    kolon_s9 = "DisKaynak_SafeTarget_EU" if tur_s9 == "EU" else "DisKaynak_SafeTarget_NonEU"
    st.dataframe(
        karsilastirma_s9[["Universite", kolon_s9, "%95 Tahmin Araligi"]],
        use_container_width=True, hide_index=True
    )

with sekme10:
    st.info(
        "🧒 **Bu sekme ne ise yarar?** 'X puan almak icin kac soru dogru yapmam lazim?' "
        "sorusuna cevap veriyor - tipki bir markette 'bu kadar param var, ne kadar sekerleme "
        "alabilirim' hesaplamak gibi. Sen bir hedef puan soyluyorsun, biz de sana 'bu kadar "
        "dogru yapman yeterli' diyoruz.\n\n"
        "**Nasil kullanilir?** Asagida hedef puanini ve kac soruyu bos birakmayi dusundugunu "
        "gir - anlik olarak gereken en az dogru sayisini goreceksin."
    )
    st.subheader("Puan Hedefi Hesaplayici")
    st.caption(
        "Denklem: 1.5 x Dogru - 0.4 x Yanlis >= Hedef Puan, ve Dogru + Yanlis + Bos = 60. "
        "Bos sorular 0 puan getirir, sadece 'oyun disi' kalirlar - riski yok ama faydasi da yok."
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        hedef_puan_secim = st.number_input("Hedef Puan (bu puan veya ustu)", min_value=0.0, max_value=90.0,
                                             value=65.0, step=0.5)
    with col_h2:
        bos_secim = st.slider("Bos birakmayi planladigin soru sayisi", 0, 60, 0)

    sonuc_hesap = gerekli_dogru_hesapla(hedef_puan_secim, bos_secim)
    if sonuc_hesap is None:
        st.error(
            f"🧒 Bu kadar (**{bos_secim}**) soruyu bos birakirsan, kalan sorularin HEPSINI "
            f"dogru yapsan bile **{hedef_puan_secim}** puana ulasamazsin (kalan "
            f"{60 - bos_secim} sorunun tavani: {1.5 * (60 - bos_secim):.1f} puan). "
            "Bos sayisini azalt."
        )
    else:
        d_min, yanlis_hakki, gercek_puan = sonuc_hesap
        st.success(
            f"🧒 **{hedef_puan_secim} puan** almak icin: en az **{d_min} dogru** yapman "
            f"yeterli (kalan **{yanlis_hakki} soruyu** yanlis yapsan bile sorun olmaz). "
            f"Bu durumda elde edecegin puan: **{gercek_puan}**."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Dogru (en az)", d_min)
        m2.metric("Yanlis (rahatlikla)", yanlis_hakki)
        m3.metric("Bos", bos_secim)

    st.divider()
    st.markdown("### Butun Senaryolar: Bos = 0 iken, her 'yanlis sayisi' icin gereken en az dogru")
    st.caption(
        "🧒 Bu tablo sana 'ne kadar cok yanlis yaparsan, o kadar cok dogruya ihtiyacin olur' "
        "iliskisini gosterir - yanlis sayin arttikca dogru ihtiyacin da artar, cunku her "
        "yanlis seni geriye cekiyor (-0.4 puan)."
    )
    tablo_satirlari = yanlis_sweep_tablosu(hedef_puan_secim)
    if tablo_satirlari:
        df_hedef = pd.DataFrame(tablo_satirlari)
        st.dataframe(df_hedef, use_container_width=True, hide_index=True)

        st.markdown("#### Yanlis Sayisi arttikca Gereken Dogru Sayisi nasil degisiyor?")
        st.line_chart(df_hedef.set_index("Yanlis")[["Gereken En Az Dogru"]])
    else:
        st.warning(f"{hedef_puan_secim} puanina HICBIR yanlis-dogru kombinasyonuyla ulasilamiyor (60 sorunun hepsi dogru olsa {1.5*60:.1f} puan, bu senin hedefinden dusuk).")

    st.divider()
    st.divider()
    st.markdown("### 🎲 Bilmedigin Sorulari Tahmin Etmeli misin? (Binom Dagilimi)")
    st.info(
        "🧒 **Bu ne ise yarar?** Sinavda bazi sorularda hicbir fikrin olmaz ama bazi "
        "siklari eleyebilirsin. Bu, 'zar atmaya deger mi' sorusuna cevap veriyor - "
        "tipki bir torbadan renkli toplar cekip, hangi renklerin daha cok oldugunu "
        "bilerek tahmin yapmak gibi."
    )
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        n_soru_secim = st.slider("Kac soruda kararsizsin (hic fikrin yok ama 1+ sikki eleyebiliyorsun)?",
                                   0, 60, 10)
    with col_b2:
        elenebilen_secim = st.select_slider("Kac sikki eleyebiliyorsun? (IMAT'ta 5 sik var)",
                                              options=[0, 1, 2, 3, 4], value=0,
                                              format_func=lambda x: f"{x} sik eleyebiliyorum (kalan {5-x} sikdan biri dogru)")

    if n_soru_secim > 0:
        sonuc_binom = sansa_birakma_analizi(n_soru_secim, elenebilen_secim)
        p = sonuc_binom["p"]
        ev_soru = sonuc_binom["ev_soru"]
        if ev_soru > 0:
            st.success(
                f"🧒 Dogru olma ihtimalin: **%{p*100:.0f}**. Her boyle soruda ORTALAMA "
                f"**+{ev_soru}** puan kazanirsin (kaybetmezsin!). {n_soru_secim} soruda "
                f"toplam beklenen kazancin: **+{sonuc_binom['ev_toplam']}** puan. "
                f"**Tahmin etmeye DEGER.**"
            )
        elif ev_soru < 0:
            st.error(
                f"🧒 Dogru olma ihtimalin: **%{p*100:.0f}**. Her boyle soruda ORTALAMA "
                f"**{ev_soru}** puan KAYBEDERSIN. {n_soru_secim} soruda toplam beklenen "
                f"kaybin: **{sonuc_binom['ev_toplam']}** puan. **Bos birakmak daha guvenli.**"
            )
        else:
            st.warning("🧒 Tam olarak basa bas - ne kar ne zarar, fark etmez.")

        st.caption(
            f"P(bu {n_soru_secim} sorudan toplamda ZARAR ETMEME ihtimali) = "
            f"**%{sonuc_binom['olumlu_olasilik']*100:.1f}** (yani tahmin ettiginde kotu "
            f"sansa da denk gelebilirsin - bu, o riskin ne kadar oldugunu gosteriyor)"
        )

        with st.expander("Olasilik Dagilimi Tablosu ve Grafigi (Binom Dagilimi)"):
            df_binom = pd.DataFrame(sonuc_binom["dagilim"])
            st.dataframe(df_binom, use_container_width=True, hide_index=True)
            st.bar_chart(df_binom.set_index("Dogru Sayisi (X)")[["Olasilik"]])

        with st.expander("🧒 5 sikta HICBIR seyi eleyemezsen ne olur? (ilginc bir sonuc)"):
            kontrol = sansa_birakma_analizi(1, 0)
            st.markdown(
                f"5 siktan hicbirini eleyemiyorsan, dogru olma ihtimalin %20 (1/5). "
                f"Beklenen deger: 0.2×1.5 + 0.8×(-0.4) = **{kontrol['ev_soru']}** - yani "
                "KUCUK bir NEGATIF sayi! Yani IMAT'ta 'hicbir fikrin yoksa' bos birakmak "
                "istatistiksel olarak tahmin etmekten biraz daha iyidir. Ama SADECE 1 "
                "sikki bile eleyebilirsen (4 kalir, %25 sans), beklenen deger pozitife "
                "doner - o zaman tahmin etmeye deger."
            )

    st.divider()
    with st.expander("🧒 Neden 'bos birakmak' zararsiz ama 'yanlis yapmak' zararli?"):
        st.markdown(
            "Bos biraktigin bir soru sana 0 puan verir - ne kazanirsin ne kaybedersin, "
            "tipki o soruyu hic gormemis gibisin. Ama yanlis cevap verirsen -0.4 puan "
            "KAYBEDERSIN - yani o soruyu yapmasaydin daha iyi olurdu. Bu yuzden EGER hic "
            "fikrin yoksa (5 siktan hicbirini eleyemiyorsan) bos birakmak matematiksel "
            "olarak daha guvenlidir. Ama en az 2 sikki eleyebiliyorsan, rastgele tahminin "
            "ortalama getirisi pozitife doner (bu, sana gonderdigin baska bir belgede "
            "gecen 'Beklenen Deger' yontemiyle ayni mantik) - o durumda tahmin etmek "
            "mantiklidir."
        )

st.divider()
st.caption("Temel veri, 2022-2025 IMAT capraz kaynak dogrulamasindan alinmistir (Testbusters, Locomotive, Futura, "
           "Wauniversity). 2024/2025 EU gecerli aday sayilari (5.609 / 7.202) ve taban puanlar kullanicinin "
           "verdigi kaynaklara gore sabitlenmistir. 2026 kontenjanlari Decreto Ministeriale n. 1005 (6 Agustos 2026) "
           "kararindan alinmistir. 2022 EU gecerli aday sayisi ve 2026 sinav sonuclari henuz bilinmiyor.")
