"""etf_panel.py - Fon Analiz Paneli için ETF ve ETF Analiz sekmeleri (yfinance, ücretsiz).

app.py içindeki mevcut fonksiyonları (line_chart, efficiency_table, drawdown_table ...)
yeniden kullanır; bu yüzden app.py'nin globals() sözlüğünü parametre olarak alır.
"""
import numpy as np
import pandas as pd
import streamlit as st

# ticker -> (ad, kategori). Buradan istediğiniz ETF'i ekleyip çıkarabilirsiniz.
ETF_EVREN = {
    "SPY": ("SPDR S&P 500", "ABD hisse"),
    "VOO": ("Vanguard S&P 500", "ABD hisse"),
    "VTI": ("Vanguard Total US Market", "ABD hisse"),
    "QQQ": ("Invesco Nasdaq-100", "ABD hisse"),
    "IWM": ("iShares Russell 2000", "ABD hisse"),
    "DIA": ("SPDR Dow Jones", "ABD hisse"),
    "SCHD": ("Schwab US Dividend Equity", "ABD hisse"),
    "XLK": ("Teknoloji", "Sektör"),
    "XLF": ("Finans", "Sektör"),
    "XLE": ("Enerji", "Sektör"),
    "XLV": ("Sağlık", "Sektör"),
    "XLY": ("Tüketici (isteğe bağlı)", "Sektör"),
    "XLP": ("Tüketici (temel)", "Sektör"),
    "XLI": ("Sanayi", "Sektör"),
    "XLU": ("Kamu hizmetleri", "Sektör"),
    "SMH": ("Yarı iletken", "Sektör"),
    "EFA": ("iShares MSCI EAFE", "Uluslararası"),
    "VEA": ("Vanguard Gelişmiş Piyasalar", "Uluslararası"),
    "EEM": ("iShares MSCI Gelişen Piyasalar", "Uluslararası"),
    "VWO": ("Vanguard Gelişen Piyasalar", "Uluslararası"),
    "TUR": ("iShares MSCI Türkiye", "Uluslararası"),
    "AGG": ("iShares Core US Aggregate Bond", "Tahvil"),
    "BND": ("Vanguard Total Bond", "Tahvil"),
    "TLT": ("iShares 20+ Yıl ABD Tahvil", "Tahvil"),
    "IEF": ("iShares 7-10 Yıl ABD Tahvil", "Tahvil"),
    "LQD": ("iShares Yatırım Yapılabilir Kurumsal", "Tahvil"),
    "HYG": ("iShares Yüksek Getirili Kurumsal", "Tahvil"),
    "GLD": ("SPDR Altın", "Emtia"),
    "IAU": ("iShares Altın", "Emtia"),
    "SLV": ("iShares Gümüş", "Emtia"),
    "USO": ("ABD Petrol Fonu", "Emtia"),
    "DBC": ("Invesco Emtia Endeksi", "Emtia"),
    "VNQ": ("Vanguard Gayrimenkul", "Gayrimenkul"),
    "IBIT": ("iShares Bitcoin Trust", "Kripto"),
}


# ---------- Veri ----------
@st.cache_data(ttl=3600, show_spinner="ETF verisi çekiliyor...")
def load_etf(tickers, start, end):
    """Düzeltilmiş (temettü dahil) kapanış fiyatları. Boşluklar doldurulmaz."""
    import yfinance as yf
    end_x = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(list(tickers), start=str(start), end=end_x, auto_adjust=True,
                     progress=False, threads=True)
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance veri döndürmedi (geçici hız sınırı olabilir)")
    close = df["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all").dropna(axis=1, how="all")
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close.index.name = "date"
    return close.astype(float)


def _to_tl(df, start, end):
    kur = load_etf(("USDTRY=X",), start, end).iloc[:, 0]
    kur = kur.reindex(df.index.union(kur.index)).ffill().reindex(df.index).bfill()
    return df.mul(kur, axis=0)


def _fiyat(tick, para, start, end):
    ep = load_etf(tuple(tick), start, end)
    eksik = [t for t in tick if t not in ep.columns]
    if eksik:
        st.warning("Veri bulunamayan ticker: " + ", ".join(eksik))
    if para == "TL":
        ep = _to_tl(ep, start, end)
    return ep


# ---------- Seçim ----------
def _evren():
    uni = dict(ETF_EVREN)
    for t in st.session_state.get("etf_ozel", []):
        uni.setdefault(t, (t, "Özel"))
    return uni


def _sec(prefix, default):
    uni = _evren()
    cats = sorted({v[1] for v in uni.values()})
    c1, c2 = st.columns([3, 2])
    picked = c1.multiselect("ETF", list(uni), default=[d for d in default if d in uni],
                            format_func=lambda t: f"{t} · {uni[t][0]}", key=f"{prefix}_pick")
    cat_sel = c2.multiselect("Kategoriye göre ekle", cats, key=f"{prefix}_cat")
    by_cat = [t for t in uni if uni[t][1] in cat_sel]
    return list(dict.fromkeys(picked + by_cat))


# ---------- Sekme: ETF ----------
def render_etf(g):
    th, start, end = g["th"], g["start"], g["end"]
    st.subheader("ETF Takip")
    st.caption("Kaynak: Yahoo Finance (yfinance). Düzeltilmiş kapanış fiyatları (temettü dahil), "
               "veriler gecikmeli olabilir ve 1 saat önbelleğe alınır.")
    with st.expander("Kendi ETF'lerini ekle"):
        giris = st.text_input("Ticker'ları virgülle yaz (örn. VGT, ARKK, EWZ)", key="etf_ozel_input")
        st.session_state["etf_ozel"] = [x.strip().upper() for x in giris.split(",") if x.strip()]

    tick = _sec("etf", ["SPY", "QQQ", "GLD", "TLT"])
    para = st.radio("Para birimi", ["USD", "TL"], horizontal=True, key="etf_para",
                    help="TL seçilirse fiyatlar USD/TRY kuruyla çarpılır; TEFAS fonlarıyla aynı baza gelir.")
    if not tick:
        st.info("En az bir ETF seç.")
        return
    try:
        ep = _fiyat(tick, para, start, end)
    except Exception as e:
        st.error(f"ETF verisi alınamadı: {e}")
        return
    if ep.empty:
        st.warning("Seçilen aralıkta veri yok.")
        return
    epf = ep.ffill()

    st.markdown(g["ticker_html"](ep), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### Fiyat performansı (başlangıç = 100, {para})")
        base = ep.apply(lambda s: s.dropna().iloc[0])
        st.altair_chart(g["line_chart"](epf / base * 100, th, "", ",.0f"),
                        use_container_width=True, theme=None)
    with c2:
        st.markdown("##### 1 aylık getiri")
        prt = g["period_returns"](ep)
        bm = prt["1 Ay"].dropna().astype(float).rename("Getiri").rename_axis("Fon").reset_index()
        if len(bm):
            st.altair_chart(g["bar_pos_neg"](bm, "Fon", "Getiri", "", th),
                            use_container_width=True, theme=None)

    st.markdown("##### Dönemsel getiri tablosu ve sıralama")
    st.dataframe(prt.style.format({c: "{:+.2%}" for c in prt.columns if c != "Sıra (3 Ay)"},
                                  na_rep="-"), use_container_width=True)
    st.download_button("Tabloyu indir (CSV)", prt.to_csv().encode("utf-8-sig"),
                       "etf_getiri.csv", "text/csv")
    mr = g["monthly_returns"](ep)
    if len(mr):
        st.markdown("##### Aylık getiri ısı haritası")
        st.altair_chart(g["monthly_heatmap"](mr, th), use_container_width=True, theme=None)


# ---------- Sekme: ETF Analiz ----------
def render_etf_analiz(g):
    th, start, end, rf, alpha = g["th"], g["start"], g["end"], g["rf"], g["alpha"]
    st.subheader("ETF Analiz")
    tick = _sec("etfan", ["SPY", "QQQ", "TLT", "GLD", "EEM"])
    o1, o2 = st.columns(2)
    para = o1.radio("Para birimi", ["USD", "TL"], horizontal=True, key="etfan_para")
    tefas_dahil = o2.checkbox("Panelde seçili TEFAS fonlarını da dahil et (TL bazında)",
                              key="etfan_tefas",
                              help="ETF'ler TL'ye çevrilir; TEFAS fonlarıyla aynı tabloda kıyaslanır.")
    if not tick:
        st.info("En az bir ETF seç.")
        return
    if tefas_dahil:
        para = "TL"
    try:
        ap = _fiyat(tick, para, start, end)
    except Exception as e:
        st.error(f"ETF verisi alınamadı: {e}")
        return
    if tefas_dahil:
        ap = pd.concat([ap, g["p"]], axis=1).sort_index()
    ap = ap.loc[:, ap.notna().sum() > 1]
    if ap.shape[1] < 2:
        st.info("Analiz için en az iki seri gerekir.")
        return
    apf = ap.ffill()

    cols = list(ap.columns)
    bench = st.selectbox("Kıyas serisi", cols, index=cols.index("SPY") if "SPY" in cols else 0,
                         key="etfan_bench")
    st.caption(f"Para birimi: {para} · Risksiz oran / MAR: %{rf * 100:.2f} · Kıyas: {bench} · "
               f"Aralık: {ap.index[0]:%d.%m.%Y} – {ap.index[-1]:%d.%m.%Y}")

    pt = st.tabs(["Risk Ayarlı Getiri", "Kıyasa Göre", "Düşüş (Drawdown)", "Oynaklık",
                  "Kuyruk Riski ve VaR", "Korelasyon", "Ölçüt Sözlüğü"])

    with pt[0]:
        et = g["efficiency_table"](ap, rf)
        if et.empty:
            st.info("Bu tablo için seri başına en az 30 gözlem gerekir.")
        else:
            efmt = {c: "{:+.2%}" for c in ["Dönem getirisi", "Yıllıklandırılmış getiri (CAGR)"]}
            efmt.update({"Yıllık volatilite (σ)": "{:.2%}", "Pozitif gün oranı": "{:.1%}"})
            efmt.update({c: "{:.2f}" for c in ["Sharpe oranı", "Sortino oranı", "Calmar oranı",
                                               "Omega oranı", "Kazanç/Kayıp oranı"]})
            st.dataframe(et.style.format(efmt, na_rep="-"), use_container_width=True)
            st.download_button("Tabloyu indir (CSV)", et.to_csv().encode("utf-8-sig"),
                               "etf_risk_ayarli.csv", "text/csv")
        rt = g["risk_table"](ap, bench, rf, alpha)
        if not rt.empty:
            st.markdown("##### Risk / getiri haritası")
            sc = (rt[["Yıllık volatilite", "Yıllık getiri"]].astype(float)
                  .rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
                  .rename_axis("Fon").reset_index())
            st.altair_chart(g["scatter_chart"](sc, th, ""), use_container_width=True, theme=None)

    with pt[1]:
        rel = g["relative_table"](ap, bench)
        if rel is not None and len(rel):
            st.markdown(f"##### Dönemsel getiri farkı (seri − {bench}), yüzde puan")
            st.dataframe(rel.style.format("{:+.2%}", na_rep="-"), use_container_width=True)
        st.markdown(f"##### Aktif performans göstergeleri (kıyas: {bench})")
        at = g["active_table"](ap, bench, rf)
        if at.empty:
            st.info("Yeterli ortak gözlem yok.")
        else:
            afmt = {"Beta (β)": "{:.2f}", "Jensen alfası (α, yıllık)": "{:+.2%}",
                    "Korelasyon (ρ)": "{:.2f}", "Belirlilik katsayısı (R²)": "{:.2f}",
                    "Aktif getiri (yıllık)": "{:+.2%}", "Takip hatası (TE)": "{:.2%}",
                    "Bilgi oranı (IR)": "{:.2f}", "Treynor oranı": "{:.2f}",
                    "Yukarı yakalama": "{:.2f}", "Aşağı yakalama": "{:.2f}"}
            st.dataframe(at.style.format(afmt, na_rep="-"), use_container_width=True)

    with pt[2]:
        st.markdown("##### Su altı grafiği: zirveden düşüş")
        st.altair_chart(g["line_chart"](apf / apf.cummax() - 1, th, "", ".0%"),
                        use_container_width=True, theme=None)
        dtb = g["drawdown_table"](ap)
        if not dtb.empty:
            dfmt = {"Maks. düşüş (MDD)": "{:.2%}", "Ortalama düşüş": "{:.2%}",
                    "Güncel düşüş": "{:.2%}", "Ulcer endeksi": "{:.2%}",
                    "Pain endeksi": "{:.2%}", "Zirveden dibe (gün)": "{:,.0f}",
                    "En uzun su altı süresi (gün)": "{:,.0f}"}
            st.dataframe(dtb.style.format(dfmt, na_rep="-"), use_container_width=True)

    with pt[3]:
        vt = g["volatility_table"](ap, rf)
        if vt.empty:
            st.info("Bu tablo için seri başına en az 30 gözlem gerekir.")
        else:
            st.dataframe(vt.style.format(
                {c: ("{:.2f}" if "tarihsel" in c else "{:.2%}") for c in vt.columns},
                na_rep="-"), use_container_width=True)
        st.markdown("##### Kayan 30 gözlemlik volatilite (yıllıklandırılmış)")
        Rv = g["returns_df"](ap)
        rvv = Rv.rolling(30, min_periods=20).std() * np.sqrt(
            pd.Series({c: g["obs_per_year"](ap[c]) for c in ap}))
        st.altair_chart(g["line_chart"](rvv, th, "", ".0%"), use_container_width=True, theme=None)

    with pt[4]:
        amt = st.number_input("Risk tutarı için yatırım tutarı", 1000.0, 1e12, 100000.0, 1000.0,
                              key="etfan_amt")
        st.markdown(f"##### Dağılım şekli ve VaR / CVaR, %{int(alpha * 100)} güven")
        tt = g["tail_table"](ap, alpha, amt)
        if tt.empty:
            st.info("Bu tablo için seri başına en az 30 gözlem gerekir.")
        else:
            tfm = {c: "{:.2%}" for c in tt.columns if c not in
                   ("Çarpıklık (S)", "Aşırı basıklık (K)", "Jarque–Bera p-değeri",
                    "Normallik (%5)", "Tarihsel VaR (TL)")}
            tfm.update({"Çarpıklık (S)": "{:.2f}", "Aşırı basıklık (K)": "{:.2f}",
                        "Jarque–Bera p-değeri": "{:.4f}", "Tarihsel VaR (TL)": "{:,.0f}"})
            st.dataframe(tt.style.format(tfm, na_rep="-"), use_container_width=True)
            st.caption(f"'Tarihsel VaR (TL)' sütunu seçilen para biriminde ({para}) tutar gösterir.")

    with pt[5]:
        st.markdown("##### Korelasyon matrisi (günlük getiriler)")
        st.altair_chart(g["corr_heatmap"](g["returns_df"](ap).corr(), th),
                        use_container_width=True, theme=None)
        if tefas_dahil:
            st.caption("ABD ve Türkiye işlem günleri farklı olduğundan ETF–TEFAS korelasyonları "
                       "ortak gözlem tarihlerine dayanır ve yaklaşık değerdir.")

    with pt[6]:
        st.markdown("##### Performans ölçütleri")
        g["show_glossary"](g["PERF_GLOSSARY"])
        st.markdown("##### Risk ölçütleri")
        g["show_glossary"](g["RISK_GLOSSARY"])

    st.caption("Geçmiş performans gelecekteki getirinin garantisi değildir; yatırım tavsiyesi değildir. "
               "Yahoo Finance verisi resmi olmayan bir kaynaktır, kritik rakamları teyit ediniz.")


# ---------- Sekme: ETF Detay (portföy içeriği) ----------
SEKTOR_TR = {
    "technology": "Teknoloji", "financial_services": "Finans", "healthcare": "Sağlık",
    "consumer_cyclical": "Tüketici (isteğe bağlı)", "consumer_defensive": "Tüketici (temel)",
    "industrials": "Sanayi", "energy": "Enerji", "utilities": "Kamu hizmetleri",
    "basic_materials": "Temel malzemeler", "communication_services": "İletişim",
    "realestate": "Gayrimenkul",
}
VARLIK_TR = {
    "stockPosition": "Hisse senedi", "bondPosition": "Tahvil", "cashPosition": "Nakit",
    "preferredPosition": "İmtiyazlı hisse", "convertiblePosition": "Dönüştürülebilir",
    "otherPosition": "Diğer",
}


@st.cache_data(ttl=21600, show_spinner="ETF portföy içeriği çekiliyor...")
def load_etf_detay(ticker):
    """Yahoo Finance fon verisi: ilk hisseler, sektör, varlık sınıfı, gider oranı."""
    import yfinance as yf
    t = yf.Ticker(ticker)

    def safe(f):
        try:
            return f()
        except Exception:
            return None

    fd = safe(lambda: t.funds_data)
    return {
        "info": safe(lambda: t.info) or {},
        "holdings": safe(lambda: fd.top_holdings) if fd is not None else None,
        "sector": safe(lambda: fd.sector_weightings) if fd is not None else None,
        "assets": safe(lambda: fd.asset_classes) if fd is not None else None,
        "overview": safe(lambda: fd.fund_overview) if fd is not None else None,
        "ops": safe(lambda: fd.fund_operations) if fd is not None else None,
    }


def _pct_table(rows, mapping):
    tbl = pd.DataFrame({t: {mapping.get(k, k): float(v) * 100 for k, v in d.items()
                            if v not in (None, 0)} for t, d in rows.items() if d}).T
    tbl = tbl.fillna(0.0)
    return tbl.loc[:, (tbl != 0).any()] if not tbl.empty else tbl


def render_etf_detay(g):
    th = g["th"]
    st.subheader("ETF Detay: portföy içeriği")
    st.caption("Kaynak: Yahoo Finance fon verisi. İlk 10 hisse, sektör ve varlık dağılımı, "
               "gider oranı ve fon büyüklüğü. Her ETF için veri bulunmayabilir "
               "(altın, kripto ve emtia ETF'lerinde hisse listesi olmaz). 6 saat önbelleğe alınır.")
    tick = _sec("etfdt", ["SPY", "QQQ"])
    if not tick:
        st.info("En az bir ETF seç.")
        return
    uni = _evren()
    data = {t: load_etf_detay(t) for t in tick}

    # 1) Genel bilgi
    gen = {}
    for t, d in data.items():
        info, ov, ops = d["info"], d["overview"] or {}, d["ops"]
        gider = np.nan
        try:
            if ops is not None and "Annual Report Expense Ratio" in ops.index:
                gider = float(ops.loc["Annual Report Expense Ratio"].iloc[0])
        except Exception:
            pass
        aum = info.get("totalAssets")
        gen[t] = {
            "Ad": info.get("longName") or info.get("shortName") or uni.get(t, (t,))[0],
            "Kategori": ov.get("categoryName") or info.get("category") or "-",
            "Fon ailesi": ov.get("family") or info.get("fundFamily") or "-",
            "Fon büyüklüğü (mlr USD)": aum / 1e9 if aum else np.nan,
            "Gider oranı": gider,
            "Temettü verimi": info.get("yield") or info.get("trailingAnnualDividendYield") or np.nan,
        }
    gt = pd.DataFrame(gen).T
    for c in ["Fon büyüklüğü (mlr USD)", "Gider oranı", "Temettü verimi"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")
    st.markdown("##### Genel bilgi")
    st.dataframe(gt.style.format({"Fon büyüklüğü (mlr USD)": "{:,.2f}", "Gider oranı": "{:.2%}",
                                  "Temettü verimi": "{:.2%}"}, na_rep="-"),
                 use_container_width=True)

    # 2) Varlık sınıfı ve sektör
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Varlık sınıfı dağılımı (%)")
        at = _pct_table({t: d["assets"] for t, d in data.items()}, VARLIK_TR)
        if at.empty:
            st.info("Varlık sınıfı verisi bulunamadı.")
        else:
            st.altair_chart(g["alloc_bar"](at, th), use_container_width=True, theme=None)
    with c2:
        st.markdown("##### Sektör dağılımı (%)")
        sc = _pct_table({t: d["sector"] for t, d in data.items()}, SEKTOR_TR)
        if sc.empty:
            st.info("Sektör verisi bulunamadı (tahvil/emtia ETF'lerinde olmayabilir).")
        else:
            st.altair_chart(g["alloc_bar"](sc, th), use_container_width=True, theme=None)

    # 3) İlk 10 hisse
    st.markdown("##### İlk 10 pozisyon")
    secili = st.selectbox("ETF", tick, key="etfdt_one")
    h = data[secili]["holdings"]
    if h is None or len(h) == 0:
        st.info(f"{secili} için pozisyon listesi bulunamadı.")
    else:
        h = h.copy()
        wcol = "Holding Percent" if "Holding Percent" in h.columns else h.columns[-1]
        h[wcol] = pd.to_numeric(h[wcol], errors="coerce")
        h = h.rename(columns={wcol: "Ağırlık", "Name": "Ad"})
        h.index.name = "Sembol"
        m1, m2 = st.columns(2)
        m1.metric("İlk 10 pozisyon toplamı", f"{h['Ağırlık'].sum():.1%}")
        m2.metric("En büyük pozisyon", f"{h['Ağırlık'].max():.1%}")
        st.dataframe(h.style.format({"Ağırlık": "{:.2%}"}, na_rep="-"), use_container_width=True)
        dn = h.reset_index()[["Sembol", "Ağırlık"]].rename(columns={"Ağırlık": "Deger"})
        st.altair_chart(g["donut_chart"](dn, "Sembol", "Deger", th, height=300),
                        use_container_width=True, theme=None)

    # 4) Ortak pozisyonlar
    if len(tick) >= 2:
        st.markdown("##### Ortak pozisyonlar (ilk 10 içinde)")
        cols = {}
        for t in tick:
            hh = data[t]["holdings"]
            if hh is not None and len(hh):
                wc = "Holding Percent" if "Holding Percent" in hh.columns else hh.columns[-1]
                cols[t] = pd.to_numeric(hh[wc], errors="coerce")
        if len(cols) >= 2:
            ov = pd.DataFrame(cols)
            ov = ov[ov.notna().sum(axis=1) >= 2].sort_index()
            if ov.empty:
                st.caption("Seçilen ETF'lerin ilk 10 pozisyonunda ortak hisse yok.")
            else:
                ov.index.name = "Sembol"
                st.dataframe(ov.style.format("{:.2%}", na_rep="-"), use_container_width=True)
    st.caption("Yatırım tavsiyesi değildir. Yahoo verisi resmi olmayan bir kaynaktır; "
               "güncel içerik için fonun resmi sayfasını kontrol ediniz.")


# ---------- Portföy sepeti için yardımcılar ----------
def etf_secenekleri():
    """ticker -> (ad, kategori) sözlüğü (kendi eklenen ticker'lar dahil)."""
    return _evren()


def etf_tl_fiyat(tick, start, end):
    """Seçilen ETF'lerin TL bazında düzeltilmiş kapanış fiyatları."""
    return _fiyat(list(tick), "TL", start, end)


def etf_gider_orani(ticker):
    """Yıllık gider oranı (oran olarak, örn. 0.0009). Bulunamazsa None."""
    try:
        ops = load_etf_detay(ticker)["ops"]
        if ops is not None and "Annual Report Expense Ratio" in ops.index:
            v = float(ops.loc["Annual Report Expense Ratio"].iloc[0])
            if v == v:
                return v
    except Exception:
        pass
    return None
