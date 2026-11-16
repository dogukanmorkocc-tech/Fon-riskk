"""veri_panel.py - Tüm TEFAS ve ETF verisi: veri katmanı + 'Tüm veriler' sekmesi.

TEFAS: tüm yatırım fonlarının bugünkü ve 1 yıl önceki fiyatı iki tarihte tek seferde alınır (1 yıllık getiri).
ETF: etf_panel.ETF_EVREN listesindeki tüm ETF'ler, TL'ye çevrilmiş olarak alınır.
Tek tek fiyat geçmişi (volatilite, düşüş, risk değeri) yalnızca ihtiyaç olunca çekilir (TEFAS hız sınırı).
"""
import numpy as np
import pandas as pd
import streamlit as st

from gorsel_panel import _ciz, _strip, stretch

KOLONLAR = ["Kod", "Ad", "Tür", "Sınıf", "Büyüklük", "Yatırımcı", "Getiri1Y", "Vol", "Sharpe", "MDD", "Risk"]


# ---------- Veri katmanı ----------
def tefas_evren(g):
    """Tüm TEFAS yatırım fonları (filtrelenmiş). Oturumda saklanır."""
    hv = st.session_state.get("tum_tefas")
    if hv is None:
        from bulten_panel import _evren
        U, uyar, ham = _evren(g)
        hv = {"U": U, "uyar": uyar, "ham": ham, "tarih": str(pd.Timestamp.today().date())}
        st.session_state["tum_tefas"] = hv
    return hv


def etf_veri(g):
    """Tüm ETF'ler: TL fiyatlar ve özet metrikler. Başarısız olursa saklanmaz (tekrar denenebilsin)."""
    hv = st.session_state.get("tum_etf")
    if hv is not None:
        return hv
    from etf_panel import ETF_EVREN
    from bulten_panel import _etf_cache
    from profil_panel import ETF_SINIF
    end = pd.Timestamp.today().normalize().date()
    start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
    tick = list(ETF_EVREN) + [t for t in ("GLD", "IAU") if t not in ETF_EVREN]
    try:
        px = _etf_cache(tuple(tick), start, end)
    except Exception as e:
        return {"px": pd.DataFrame(), "E": pd.DataFrame(columns=KOLONLAR), "hata": f"{type(e).__name__}: {e}"}
    et = g["efficiency_table"](px, g["rf"])
    dt = g["drawdown_table"](px)
    rows = []
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 30:
            continue
        ad, kat = ETF_EVREN.get(t, (t, "Altın"))
        rows.append({"Kod": t, "Ad": ad, "Tür": "ETF", "Sınıf": ETF_SINIF.get(kat, "Altın" if t in ("GLD", "IAU") else "Yabancı varlık / ETF"),
                     "Büyüklük": np.nan, "Yatırımcı": np.nan, "Getiri1Y": s.iloc[-1] / s.iloc[0] - 1,
                     "Vol": et.loc[t, "Yıllık volatilite (σ)"], "Sharpe": et.loc[t, "Sharpe oranı"],
                     "MDD": pd.to_numeric(dt.loc[t, "Maks. düşüş (MDD)"], errors="coerce"),
                     "Risk": g["risk_degeri"](s)[0]})
    hv = {"px": px, "E": pd.DataFrame(rows, columns=KOLONLAR), "hata": None}
    st.session_state["tum_etf"] = hv
    return hv


def tum_veri(g, etf=True):
    """TEFAS + ETF tek tabloda. Ek olarak çekilmiş geçmiş metrikleri (volatilite, risk değeri) birleştirir."""
    T = tefas_evren(g)["U"]
    A = pd.DataFrame({"Kod": T.index, "Ad": T["Fon adı"].values, "Tür": "TEFAS", "Sınıf": T["Sınıf"].values,
                      "Büyüklük": T["Büyüklük"].values, "Yatırımcı": T["Yatırımcı"].values,
                      "Getiri1Y": T["Getiri1Y"].values})
    for c in ("Vol", "Sharpe", "MDD", "Risk"):
        A[c] = np.nan
    ek = st.session_state.get("gecmis_m", {})
    for kod, m in ek.items():
        i = A.index[A["Kod"] == kod]
        for c, v in m.items():
            A.loc[i, c] = v
    if etf:
        E = etf_veri(g)["E"]
        if len(E):
            A = pd.concat([A, E[KOLONLAR]], ignore_index=True)
    return A


def gecmis_cek(g, kodlar):
    """Verilen TEFAS fonlarının 1 yıllık geçmişini çeker (hız sınırına uygun), metrikleri hesaplar.
    Dönüş: (fiyat DataFrame, {kod: metrik sözlüğü}, başarısız kodlar, ilk hata)."""
    from bulten_panel import _fiyat_tek
    end = pd.Timestamp.today().normalize().date()
    s0, e0 = str((pd.Timestamp(end) - pd.DateOffset(years=1)).date()), str(end)
    cols, fail, hata = {}, [], None
    prog = st.progress(0.0, text=f"{len(kodlar)} fonun geçmiş fiyatı çekiliyor (dakikada ~6 istek sınırı)...") if kodlar else None
    for i, k in enumerate(kodlar, 1):
        try:
            s = g["to_wide"](_fiyat_tek(k, s0, e0))["price"].get(k)
            if s is None or s.dropna().empty:
                raise ValueError("boş fiyat serisi")
            cols[k] = s
        except Exception as e:
            fail.append(k)
            hata = hata or f"{type(e).__name__}: {str(e)[:90]}"
        prog.progress(i / len(kodlar))
    if prog:
        prog.empty()
    px = pd.DataFrame(cols)
    ek = st.session_state.setdefault("gecmis_m", {})
    if px.shape[1]:
        et = g["efficiency_table"](px, g["rf"])
        dt = g["drawdown_table"](px)
        for k in px.columns:
            ek[k] = {"Vol": et.loc[k, "Yıllık volatilite (σ)"], "Sharpe": et.loc[k, "Sharpe oranı"],
                     "MDD": pd.to_numeric(dt.loc[k, "Maks. düşüş (MDD)"], errors="coerce"),
                     "Risk": g["risk_degeri"](px[k])[0]}
    return px, {k: ek[k] for k in px.columns}, fail, hata


# ---------- Tablo gösterimi ----------
def goster_tablo(df, yukseklik=420):
    """Ortak tablo: oranları yüzdeye çevirir, Türkçe başlıklar kullanır."""
    t = pd.DataFrame({
        "Kod": df["Kod"], "Ad": df["Ad"].astype(str).str.slice(0, 55), "Tür": df["Tür"], "Sınıf": df["Sınıf"],
        "1Y getiri (%)": (df["Getiri1Y"] * 100).round(1),
        "Büyüklük (mn TL)": (df["Büyüklük"] / 1e6).round(0),
        "Yatırımcı": df["Yatırımcı"],
        "Volatilite (%)": (df["Vol"] * 100).round(1),
        "Maks. düşüş (%)": (df["MDD"] * 100).round(1),
        "Risk (1-7)": df["Risk"]})
    stretch(st.dataframe, t, hide_index=True, height=yukseklik)
    return t


# ---------- 'Tüm veriler' sekmesi ----------
def render_veri(g):
    th, pc = g["th"], g["pc"]
    st.markdown("### Tüm TEFAS ve ETF verileri")
    st.caption("TEFAS'taki tüm yatırım fonları (bugünkü ve 1 yıl önceki fiyattan 1 yıllık getiri, büyüklük, yatırımcı sayısı) "
               "ve ETF listesi (TL'ye çevrilmiş) tek tabloda. Tablo sıralanabilir, aranabilir ve indirilebilir. "
               "Volatilite ve risk değeri yalnızca geçmişi çekilmiş fonlar için dolu gelir (komut kutusundan fon sorarsanız dolar).")
    if not st.session_state.get("veri_yuklu"):
        st.info("İlk yükleme TEFAS hız sınırı nedeniyle 1-2 dakika sürebilir; sonuçlar 6 saat saklanır.")
        if st.button("Tüm veriyi yükle", type="primary", key="veri_btn"):
            st.session_state["veri_yuklu"] = True
            (getattr(st, "rerun", None) or st.experimental_rerun)()
        return
    with st.spinner("TEFAS ve ETF verileri hazırlanıyor..."):
        try:
            hv = tefas_evren(g)
            ev = etf_veri(g)
            A = tum_veri(g)
        except Exception as e:
            st.error(f"Veri alınamadı: {type(e).__name__}: {e}")
            st.session_state["veri_yuklu"] = False
            return
    for u in hv["uyar"]:
        st.caption(u)
    if ev.get("hata"):
        st.warning(f"ETF verisi alınamadı: {ev['hata']}. Birkaç dakika sonra sayfayı yenileyin.")
    tf, et = A[A["Tür"] == "TEFAS"], A[A["Tür"] == "ETF"]
    _strip([("TEFAS fon", f"{len(tf):,}".replace(",", "."), f"taranan {hv['ham']:,} fondan filtre sonrası".replace(",", "."), ""),
            ("ETF", str(len(et)), "TL cinsinden", ""),
            ("Medyan 1Y getiri", pc(tf["Getiri1Y"].median()), "TEFAS fonları", ""),
            ("En yüksek 1Y", pc(tf["Getiri1Y"].max()), str(tf.loc[tf["Getiri1Y"].idxmax(), "Kod"]) if tf["Getiri1Y"].notna().any() else "-", "up"),
            ("Veri tarihi", hv["tarih"], "bugün ve 1 yıl öncesi", "")])

    f1, f2, f3 = st.columns([2, 2, 3])
    tur = f1.selectbox("Tür", ["Tümü", "TEFAS", "ETF"], key="v_tur")
    siniflar = sorted(A["Sınıf"].dropna().unique())
    sec = f2.multiselect("Sınıf", siniflar, key="v_sinif")
    ara = f3.text_input("Ara (kod veya ad)", key="v_ara")
    s1, s2 = st.columns(2)
    siralama = s1.selectbox("Sırala", ["1Y getiri", "Büyüklük", "Yatırımcı", "Volatilite", "Kod"], key="v_sira")
    yon = s2.radio("Yön", ["Azalan", "Artan"], horizontal=True, key="v_yon")
    F = A.copy()
    if tur != "Tümü":
        F = F[F["Tür"] == tur]
    if sec:
        F = F[F["Sınıf"].isin(sec)]
    if ara.strip():
        a = ara.strip().upper()
        F = F[F["Kod"].str.upper().str.contains(a, regex=False) | F["Ad"].str.upper().str.contains(a, regex=False)]
    kol = {"1Y getiri": "Getiri1Y", "Büyüklük": "Büyüklük", "Yatırımcı": "Yatırımcı", "Volatilite": "Vol", "Kod": "Kod"}[siralama]
    F = F.sort_values(kol, ascending=(yon == "Artan"), na_position="last")
    st.markdown(f"##### {len(F):,} kayıt".replace(",", "."))
    t = goster_tablo(F)
    st.download_button("Tabloyu CSV indir", t.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"tefas_etf_veri_{hv['tarih']}.csv", mime="text/csv", key="v_csv")

    st.markdown("##### Sınıflara göre görünüm")
    g1, g2 = st.columns(2)
    sn = tf.groupby("Sınıf").agg(Sayi=("Kod", "count"), Medyan=("Getiri1Y", "median")).reset_index()
    with g1:
        st.caption("Sınıf başına TEFAS fon sayısı")
        d1 = sn.rename(columns={"Sınıf": "Fon", "Sayi": "Getiri"}).assign(Getiri=lambda x: x["Getiri"].astype(float))[["Fon", "Getiri"]]
        _ciz(lambda: g["bar_pos_neg"](d1, "Fon", "Getiri", "", th, ",.0f"))
    with g2:
        st.caption("Sınıf başına medyan 1 yıllık getiri")
        d2 = sn.rename(columns={"Sınıf": "Fon", "Medyan": "Getiri"})[["Fon", "Getiri"]].dropna()
        _ciz(lambda: g["bar_pos_neg"](d2, "Fon", "Getiri", "", th))
    st.markdown("##### En yüksek 15 getiri (filtrelenmiş liste)")
    top = F.dropna(subset=["Getiri1Y"]).sort_values("Getiri1Y", ascending=False).head(15)
    if len(top):
        d3 = top[["Kod", "Getiri1Y"]].rename(columns={"Kod": "Fon", "Getiri1Y": "Getiri"})
        _ciz(lambda: g["bar_pos_neg"](d3, "Fon", "Getiri", "", th))
    st.caption("Not: Getiriler geçmiş verilere dayanır ve gelecekteki getirinin garantisi değildir; bu tablo yatırım tavsiyesi değildir.")
