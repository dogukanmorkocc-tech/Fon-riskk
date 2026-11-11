"""bulten_panel.py - Fon analiz bülteni.

Akış: kullanıcı ne incelemek istediğini yazar -> altta analizleri işaretler -> bülten otomatik oluşur.
Veri: TEFAS (fonlar) ve Yahoo Finance (ETF'ler). app.py'nin globals() sözlüğünü parametre olarak alır.
"""
import html as _html_mod
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

DISC = ("Bu bülten bilgilendirme amaçlıdır; yatırım tavsiyesi değildir. Veriler geçmişe dayanır, geçmiş performans "
        "gelecekteki getirinin garantisi değildir. Karar vermeden önce fon izahnamesini ve KAP duyurularını inceleyin.")
STOP = {"VAR", "CVAR", "ETF", "SPK", "TEFAS", "KAP", "BIST", "USD", "EUR", "TRY", "FON", "ILE", "SON", "TL", "RISK",
        "VE", "VEYA", "EN", "BU", "NE", "NASIL", "HANGI", "ICIN", "GETIRI", "ILK", "BANA", "SANA", "BIR", "GIBI",
        "YIL", "AY", "HAFTA", "AYLIK", "BIN", "MN", "LIRA", "TUTAR", "DE", "DA", "SU", "YATIRIM", "ANALIZ", "FONU"}
MODULLER = [("sepet", "Portföy sepeti"), ("perf", "Performans"), ("risk", "Risk"),
            ("kars", "Yatırımcı karşılaştırma"), ("etf", "ETF analizi (Yahoo Finance)"),
            ("akis", "Fon büyüklüğü ve akış"), ("varlik", "Varlık dağılımı"), ("profil", "Yatırımcı profili uyumu")]
FMT_EFF = {"Dönem getirisi": "{:+.2%}", "Yıllıklandırılmış getiri (CAGR)": "{:+.2%}", "Yıllık volatilite (σ)": "{:.2%}",
           "Pozitif gün oranı": "{:.1%}", "Sharpe oranı": "{:.2f}", "Sortino oranı": "{:.2f}", "Calmar oranı": "{:.2f}",
           "Omega oranı": "{:.2f}", "Kazanç/Kayıp oranı": "{:.2f}"}
FMT_INV = {"Brüt fon getirisi": "{:.2%}", "Brüt dönem sonu değer (TL)": "{:,.0f}", "Brüt kâr/zarar (TL)": "{:,.0f}",
           "Stopaj oranı": "{:.2%}", "Stopaj (TL)": "{:,.0f}", "Masraf (TL)": "{:,.0f}",
           "Net elde edilen (TL)": "{:,.0f}", "Net kâr/zarar (TL)": "{:,.0f}", "Net getiri": "{:.2%}",
           "Yıllık volatilite": "{:.2%}", "Maks. düşüş": "{:.2%}", "Maks. düşüş (TL)": "{:,.0f}",
           "Kötü ay %5 (getiri)": "{:.2%}", "Kötü ay %5 (TL)": "{:,.0f}", "Sharpe": "{:.2f}",
           "Risk değeri (1-7)": "{:.0f}"}
FMT_BASKET = {"Tutar (TL)": "{:,.0f}", "Ağırlık": "{:.1%}", "Dönem getirisi": "{:+.2%}", "Risk katkısı": "{:.1%}",
              "Brüt değer (TL)": "{:,.0f}", "Stopaj (TL)": "{:,.0f}", "Masraf (TL)": "{:,.0f}",
              "Gider oranı (TL)": "{:,.0f}", "Net değer (TL)": "{:,.0f}", "Net kâr/zarar (TL)": "{:+,.0f}"}


# ---------- Metni anlama ----------
def _norm(g, s):
    return re.sub(r"[’'`´]", "", s.translate(g["TR_MAP"]).lower())


def _sayi(sn):
    s = sn.strip()
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        return float(s.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        return float(s.replace(",", ""))
    return float(s.replace(",", "."))


def _coz(g, yazi):
    """Yazıdan TEFAS fon kodları, ETF'ler, tutar ve dönem çıkarır."""
    from etf_panel import ETF_EVREN
    ozel = st.session_state.get("etf_ozel", [])
    L = "A-Za-zÇĞİÖŞÜçğıöşü"
    tr = str.maketrans("İŞĞÜÖÇ", "ISGUOC")
    known = {c.upper(): c for c in g["p"].columns}
    etw = "etf" in yazi.lower()
    tefas, etfs = [], []
    for m in re.finditer(rf"(?<![{L}0-9])([{L}0-9]{{2,5}})(?![{L}0-9])", yazi):
        tok = m.group(1)
        if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]", tok):      # yalnız rakamlar (200, 2025) kod değildir
            continue
        t = tok.upper().translate(tr)
        if t in STOP:
            continue
        if t in ETF_EVREN or t in ozel:
            if (tok.isupper() or etw) and t not in etfs:
                etfs.append(t)
        elif 3 <= len(t) <= 4 and (t in known or tok.isupper()):
            c = known.get(t, t)
            if c not in tefas:
                tefas.append(c)
    for m in re.finditer(r"\b([A-Za-z]{2,5})\s+etf\b", yazi, re.I):
        tok = m.group(1)
        t = tok.upper()
        if tok.isupper() and t not in STOP and t not in etfs:
            etfs.append(t)
            if t in tefas:
                tefas.remove(t)
    qn = _norm(g, yazi)
    tutar = None
    m = re.search(r"(\d+(?:[.,]\d+)*)\s*(bin|milyon|mn|k)?\s*(?:tl|lira|₺)", qn)
    if m:
        try:
            v = _sayi(m.group(1))
            v *= {"bin": 1e3, "k": 1e3, "milyon": 1e6, "mn": 1e6}.get(m.group(2), 1)
            tutar = v
        except Exception:
            pass
    gun = None
    for m in re.finditer(r"(?<!\d)(\d{1,3})\s*(yil|ay|hafta)", qn):
        n, birim = int(m.group(1)), m.group(2)
        if n <= {"yil": 30, "ay": 360, "hafta": 520}[birim] and n > 0:    # '2025 yılı' gibi yıl numaralarını ele
            gun = n * {"yil": 365, "ay": 30, "hafta": 7}[birim]
            break
    if gun is None and ("yilbasi" in qn or "yil basi" in qn):
        gun = (pd.Timestamp(g["end"]) - pd.Timestamp(year=pd.Timestamp(g["end"]).year, month=1, day=1)).days
    return tefas[:8], etfs[:6], tutar, gun


# ---------- Veri ----------
def _tefas_seri(g, code, start, end):
    s0, e0 = str(start), str(end)
    raw = g["load_tefas"]((code,), s0, e0) if g["live"] else g["load_synthetic"]((code,), s0, e0)
    w = g["to_wide"](raw)["price"]
    if code not in w.columns or w[code].dropna().empty:
        raise RuntimeError("veri yok")
    ad = code
    if "title" in raw.columns:
        t = raw.dropna(subset=["title"])
        if len(t):
            ad = str(t["title"].iloc[0])
    return w[code].dropna(), ad


def _veri(g, tefas, etfs, start, end):
    from etf_panel import ETF_EVREN, etf_tl_fiyat
    seri, ad, uyar, tefas_ok, etf_ok = {}, {}, [], [], list(etfs)
    for c in tefas:
        try:
            seri[c], ad[c] = _tefas_seri(g, c, start, end)
            tefas_ok.append(c)
        except Exception:
            try:    # TEFAS'ta yoksa Yahoo Finance ETF olabilir
                e = etf_tl_fiyat((c,), start, end)
                if c in e.columns:
                    seri[c] = e[c].dropna()
                    etf_ok.append(c)
                    continue
            except Exception:
                pass
            uyar.append(f"{c}: TEFAS'ta veya Yahoo Finance'ta veri bulunamadı, bültene alınmadı.")
    if etfs:
        try:
            e = etf_tl_fiyat(tuple(etfs), start, end)
            for t in e.columns:
                seri[t] = e[t].dropna()
        except Exception as ex:
            uyar.append(f"ETF verisi alınamadı: {ex}")
    etf_ok = [t for t in dict.fromkeys(etf_ok) if t in seri]
    for t in etf_ok:
        ad[t] = ETF_EVREN.get(t, (t,))[0]
    if not seri:
        return None, {}, [], [], uyar
    P = pd.concat(seri, axis=1).sort_index()
    return P, ad, tefas_ok, etf_ok, uyar


# ---------- Bölüm oluşturucular ----------
def _bos(baslik):
    return {"baslik": baslik, "maddeler": [], "tablolar": [], "grafikler": [], "uyari": []}


def sec_ozet(g, c):
    pc, rf, alpha = g["pc"], g["rf"], g["alpha"]
    s = _bos("Bülten özeti")
    P = c["P"]
    s["maddeler"] += [
        f"Kapsam: **{len(c['tefas'])} TEFAS fonu**, **{len(c['etfs'])} ETF** (Yahoo Finance, TL'ye çevrilmiş). "
        f"Dönem: {P.index[0]:%d.%m.%Y} – {P.index[-1]:%d.%m.%Y}.",
        f"Varsayımlar: risksiz oran %{rf * 100:.1f}, VaR güven düzeyi %{int(alpha * 100)}, "
        f"yatırım tutarı {g['tl'](c['tutar'])}.", ]
    pr = st.session_state.get("profil")
    if pr:
        from profil_panel import PROFILLER
        s["maddeler"].append(f"Yatırımcı profili: **{PROFILLER[pr['idx']]}** (risk seviyesi {pr.get('seviye', '-')}/7).")
    rows = {}
    for col in P.columns:
        x = P[col].dropna()
        rd = g["risk_degeri"](x)[0]
        rows[col] = {"Ad": c["ad"].get(col, col)[:50], "Tür": "ETF" if col in c["etfs"] else "TEFAS",
                     "Kategori (tahmini)": "ETF" if col in c["etfs"] else g["fund_category"](c["ad"].get(col, "")),
                     "Risk değeri (1-7)": rd, "Dönem getirisi": x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else np.nan}
    s["tablolar"].append(("İncelenen varlıklar", pd.DataFrame(rows).T,
                          {"Risk değeri (1-7)": "{:.0f}", "Dönem getirisi": "{:+.2%}"}))
    return s


def sec_perf(g, c):
    pc, rf, th = g["pc"], g["rf"], g["th"]
    P = c["P"]
    s = _bos("Performans")
    pr = g["period_returns"](P)
    et = g["efficiency_table"](P, rf)
    if not et.empty:
        tot = et["Dönem getirisi"]
        s["maddeler"].append(f"Seçilen dönemde en yüksek getiri **{tot.idxmax()}** ({pc(tot.max())}); en düşük "
                             f"**{tot.idxmin()}** ({pc(tot.min())}).")
        sh = et["Sharpe oranı"].dropna()
        if len(sh):
            s["maddeler"].append(f"Risk başına getiri (Sharpe) en yüksek **{sh.idxmax()}** ({sh.max():.2f}).")
            neg = list(sh.index[sh < 0])
            if neg:
                s["maddeler"].append("Getirisi risksiz oranın altında kalanlar: " + ", ".join(neg) + ".")
    if "3 Ay" in pr and pr["3 Ay"].notna().any():
        l3 = pr["3 Ay"].astype(float)
        s["maddeler"].append(f"Son 3 ayda lider **{l3.idxmax()}** ({pc(l3.max())}), sonuncu **{l3.idxmin()}** ({pc(l3.min())}).")
    base = P.apply(lambda x: x.dropna().iloc[0])
    s["grafikler"].append(("Fiyat gelişimi (başlangıç = 100)", g["line_chart"](P.ffill() / base * 100, th, "", ",.0f")))
    if "1 Ay" in pr and pr["1 Ay"].notna().any():
        bm = pr["1 Ay"].dropna().astype(float).rename("Getiri").rename_axis("Fon").reset_index()
        s["grafikler"].append(("1 aylık getiri", g["bar_pos_neg"](bm, "Fon", "Getiri", "", th)))
    mr = g["monthly_returns"](P)
    if len(mr):
        s["grafikler"].append(("Aylık getiri ısı haritası", g["monthly_heatmap"](mr, th)))
    s["tablolar"].append(("Dönemsel getiri", pr, {k: "{:+.2%}" for k in pr.columns if k != "Sıra (3 Ay)"}))
    if not et.empty:
        s["tablolar"].append(("Risk ayarlı getiri", et, FMT_EFF))
    return s


def sec_risk(g, c):
    pc, rf, alpha, th = g["pc"], g["rf"], g["alpha"], g["th"]
    P = c["P"]
    s = _bos("Risk")
    rt = g["risk_table"](P, P.columns[0], rf, alpha).astype(float)
    if rt.empty:
        s["uyari"].append("Risk analizi için yeterli veri yok (en az 21 gözlem).")
        return s
    rt.insert(0, "Risk değeri (1-7)", [g["risk_degeri"](P[x])[0] for x in rt.index])
    q = int(alpha * 100)
    rv = rt["Risk değeri (1-7)"].dropna()
    if len(rv):
        s["maddeler"].append("Tahmini TEFAS risk değerleri: " + ", ".join(f"{k} {int(v)}/7" for k, v in rv.items()) + ".")
    hv = rt["Yıllık volatilite"].idxmax()
    s["maddeler"].append(f"En oynak varlık **{hv}** (yıllık volatilite {pc(rt.loc[hv, 'Yıllık volatilite'])}); en büyük "
                         f"düşüş **{rt['Maks. düşüş'].idxmin()}** ({pc(rt['Maks. düşüş'].min())}).")
    vcol = f"VaR %{q} (1g)"
    if vcol in rt:
        hvar = rt[vcol].idxmax()
        s["maddeler"].append(f"%{q} güven düzeyinde 1 günlük VaR en yüksek **{hvar}** ({pc(rt.loc[hvar, vcol])}): "
                             f"geçmişte 100 işlem gününden {q} tanesinde günlük kayıp bu oranı geçmedi.")
    if P.shape[1] >= 2:
        kor = g["returns_df"](P).corr()
        cc = kor.where(~np.eye(len(kor), dtype=bool)).stack()
        if len(cc):
            a, b = cc.idxmax()
            s["maddeler"].append(f"En çok benzeşen çift **{a}** – **{b}** (korelasyon {cc.max():.2f})."
                                 + (" Birlikte tutmak çeşitlendirmeyi azaltır." if cc.max() > 0.9 else ""))
    cols = [x for x in ["Risk değeri (1-7)", "Yıllık getiri", "Yıllık volatilite", "Sharpe", "Maks. düşüş", vcol,
                        f"CVaR %{q} (1g)"] if x in rt.columns]
    fm = {x: "{:.2%}" for x in cols}
    fm.update({"Risk değeri (1-7)": "{:.0f}", "Yıllık getiri": "{:+.2%}", "Sharpe": "{:.2f}"})
    s["tablolar"].append(("Risk göstergeleri", rt[cols], fm))
    sc = (rt[["Yıllık volatilite", "Yıllık getiri"]].rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
          .rename_axis("Fon").reset_index())
    s["grafikler"].append(("Risk / getiri haritası", g["scatter_chart"](sc, th, "")))
    if P.shape[1] >= 2:
        s["grafikler"].append(("Korelasyon matrisi", g["corr_heatmap"](g["returns_df"](P).corr(), th)))
    Pf = P.ffill()
    s["grafikler"].append(("Zirveden düşüş (drawdown)", g["line_chart"](Pf / Pf.cummax() - 1, th, "", ".0%")))
    dtb = g["drawdown_table"](P)
    if not dtb.empty:
        s["tablolar"].append(("Düşüş analizi", dtb, {"Maks. düşüş (MDD)": "{:.2%}", "Güncel düşüş": "{:.2%}",
                                                     "Ortalama düşüş": "{:.2%}"}))
    s["maddeler"].append(f"VaR güven düzeyi %{q}: yaklaşık {int(round(1 / (1 - alpha)))} günde bir bu eşik aşılır. "
                         "CVaR, eşiğin aşıldığı günlerdeki ortalama kayıptır.")
    return s


def _stopajlar(g, c, cols):
    return {x: (0.0 if x in c["etfs"] else g["auto_stopaj"](c["ad"].get(x, ""))[0]) for x in cols}


def sec_sepet(g, c):
    from etf_panel import etf_gider_orani
    pc, tl, alpha, th = g["pc"], g["tl"], g["alpha"], g["th"]
    P = c["P"]
    s = _bos("Portföy sepeti")
    cols = list(P.columns)
    amounts = {x: c["tutar"] / len(cols) for x in cols}
    res = g["basket_analysis"](P, amounts, None, _stopajlar(g, c, cols), 0.0, 0.0, alpha)
    if res is None:
        s["uyari"].append("Sepet için fonların ortak tarihlerinde yeterli veri yok.")
        return s
    s["maddeler"].append(f"Tutar ({tl(c['tutar'])}) varlıklara eşit bölündü: her birine {tl(c['tutar'] / len(cols))}. "
                         "Farklı tutarlar için 'Portföy sepeti' sekmesini kullanabilirsiniz.")
    s["maddeler"] += g["basket_text"](res, "seçilen aralık")
    rv, rvol = g["risk_degeri"](res["total"])
    if not pd.isna(rv):
        s["maddeler"].append(f"Sepetin tahmini TEFAS risk değeri: **{int(rv)}/7** (haftalık getirilerden yıllık volatilite %{rvol:.1f}).")
    if c["etfs"]:
        er = {t: etf_gider_orani(t) for t in c["etfs"]}
        txt = ", ".join(f"{t} yıllık %{v * 100:.2f}" for t, v in er.items() if v is not None)
        s["maddeler"].append("ETF'ler USD/TRY ile TL'ye çevrildi; stopaj/vergi ve komisyon 0 varsayıldı. ETF gider oranı fiyata "
                             "zaten yansıdığı için ayrıca düşülmedi" + (f" ({txt})." if txt else "."))
    t = res["table"]
    s["tablolar"].append(("Sepet tablosu", t, FMT_BASKET))
    dn = t["Tutar (TL)"].rename("Deger").rename_axis("Fon").reset_index()
    s["grafikler"].append(("Tutar dağılımı", g["donut_chart"](dn, "Fon", "Deger", th)))
    s["grafikler"].append(("Sepet değeri (TL), fon bazında", g["stacked_area"](res["value"], th, "", ".2s")))
    pf = t["Net kâr/zarar (TL)"].rename("Sonuc").rename_axis("Fon").reset_index()
    s["grafikler"].append(("Fon bazında net kâr/zarar (TL)", g["bar_pos_neg"](pf, "Fon", "Sonuc", "", th, ".2s")))
    return s


def sec_kars(g, c):
    rf = g["rf"]
    P = c["P"]
    s = _bos("Yatırımcı karşılaştırma")
    cols = list(P.columns)
    if len(cols) < 2:
        s["uyari"].append("Karşılaştırma için en az iki varlık gerekir.")
        return s
    m = g["investor_metrics"](P, cols, c["tutar"], None, rf, _stopajlar(g, c, cols), 0.0, 0.0)
    if m.empty:
        s["uyari"].append("Seçilen varlıklar için yeterli veri yok.")
        return s
    s["maddeler"] += g["investor_text"](m, c["tutar"], "seçilen aralık", rf)
    s["tablolar"].append((f"{g['tl'](c['tutar'])} her birinde değerlendirilseydi", m, FMT_INV))
    return s


def sec_etf(g, c):
    from etf_panel import SEKTOR_TR, VARLIK_TR, _pct_table, load_etf_detay
    pc, rf, th = g["pc"], g["rf"], g["th"]
    s = _bos("ETF analizi (Yahoo Finance)")
    if not c["etfs"]:
        s["uyari"].append("Metinde ETF bulunamadı. Örn. 'SPY', 'QQQ', 'GLD' veya 'VGT ETF' yazın.")
        return s
    PE = c["P"][c["etfs"]]
    pr = g["period_returns"](PE)
    et = g["efficiency_table"](PE, rf)
    if not et.empty:
        t = et["Dönem getirisi"]
        s["maddeler"].append(f"ETF'ler arasında dönem getirisi en yüksek **{t.idxmax()}** ({pc(t.max())}), en düşük "
                             f"**{t.idxmin()}** ({pc(t.min())}). Getiriler USD/TRY ile TL'ye çevrilmiştir.")
    s["tablolar"].append(("ETF dönemsel getiri (TL)", pr, {k: "{:+.2%}" for k in pr.columns if k != "Sıra (3 Ay)"}))
    if not et.empty:
        et2 = et.copy()
        et2.insert(0, "Risk değeri (1-7)", [g["risk_degeri"](PE[x])[0] for x in et2.index])
        s["tablolar"].append(("ETF risk ayarlı getiri", et2, {**FMT_EFF, "Risk değeri (1-7)": "{:.0f}"}))
    data = {}
    for t in c["etfs"]:
        try:
            data[t] = load_etf_detay(t)
        except Exception:
            data[t] = {"info": {}, "holdings": None, "sector": None, "assets": None, "overview": None, "ops": None}
    gen = {}
    for t, d in data.items():
        info, ov, ops = d["info"] or {}, d["overview"] or {}, d["ops"]
        gider = np.nan
        try:
            if ops is not None and "Annual Report Expense Ratio" in ops.index:
                gider = float(ops.loc["Annual Report Expense Ratio"].iloc[0])
        except Exception:
            pass
        aum = info.get("totalAssets")
        gen[t] = {"Ad": info.get("longName") or info.get("shortName") or c["ad"].get(t, t),
                  "Kategori": ov.get("categoryName") or info.get("category") or "-",
                  "Fon büyüklüğü (mlr USD)": aum / 1e9 if aum else np.nan, "Gider oranı": gider,
                  "Temettü verimi": info.get("yield") or info.get("trailingAnnualDividendYield") or np.nan}
    gt = pd.DataFrame(gen).T
    for k in ["Fon büyüklüğü (mlr USD)", "Gider oranı", "Temettü verimi"]:
        gt[k] = pd.to_numeric(gt[k], errors="coerce")
    s["tablolar"].append(("ETF genel bilgi", gt, {"Fon büyüklüğü (mlr USD)": "{:,.2f}", "Gider oranı": "{:.2%}",
                                                  "Temettü verimi": "{:.2%}"}))
    at = _pct_table({t: d["assets"] for t, d in data.items()}, VARLIK_TR)
    if not at.empty:
        s["grafikler"].append(("ETF varlık sınıfı dağılımı (%)", g["alloc_bar"](at, th)))
    sc = _pct_table({t: d["sector"] for t, d in data.items()}, SEKTOR_TR)
    if not sc.empty:
        s["grafikler"].append(("ETF sektör dağılımı (%)", g["alloc_bar"](sc, th)))
    for t, d in data.items():
        h = d["holdings"]
        if h is not None and len(h):
            h = h.copy()
            wc = "Holding Percent" if "Holding Percent" in h.columns else h.columns[-1]
            h[wc] = pd.to_numeric(h[wc], errors="coerce")
            h = h.rename(columns={wc: "Ağırlık", "Name": "Ad"})
            h.index.name = "Sembol"
            s["maddeler"].append(f"{t}: ilk 10 pozisyonun toplam payı **{pc(h['Ağırlık'].sum())}**, en büyük pozisyon "
                                 f"{h['Ağırlık'].idxmax()} ({pc(h['Ağırlık'].max())}).")
            s["tablolar"].append((f"{t}: ilk 10 pozisyon", h, {"Ağırlık": "{:.2%}"}))
    s["maddeler"].append("ETF içerik verisi Yahoo Finance'ten gelir; her ETF için bulunmayabilir ve gecikmeli olabilir.")
    return s


def sec_akis(g, c):
    pc, tl, th = g["pc"], g["tl"], g["th"]
    s = _bos("Fon büyüklüğü ve akış")
    if not c["tefas"]:
        s["uyari"].append("Bu bölüm TEFAS fonları içindir; metinde fon kodu yazın.")
        return s
    if not g["live"]:
        s["uyari"].append("Canlı TEFAS verisi kapalı; bu bölüm için yan panelde açın.")
        return s
    cur, ref, cur_d, ref_d = g["load_fund_info"](str(pd.Timestamp.today().date()))
    ft = g["fund_table"](cur, ref)
    kodlar = [x for x in c["tefas"] if x in ft.index]
    if not kodlar:
        s["uyari"].append("Seçilen fonlar TEFAS fon bilgisinde bulunamadı.")
        return s
    sm = g["selection_summary"](ft, kodlar)
    s["maddeler"].append(f"Seçilen {sm['n']} fonun toplam büyüklüğü {sm['size'] / 1e9:,.2f} mlr TL (piyasa payı "
                         f"{pc(sm['market_share'])}), yatırımcı sayısı {sm['investors']:,.0f} (veri tarihi {cur_d}).")
    if not pd.isna(sm["flow"]):
        s["maddeler"].append(f"Karşılaştırma tarihinden ({ref_d}) bu yana tahmini toplam net akış {sm['flow'] / 1e6:+,.0f} mn TL "
                             "(pay sayısı değişimi × güncel fiyat, yaklaşık).")
    tb = sm["table"].copy()
    cols = [x for x in ["Fon adı", "Kategori", "Portföy büyüklüğü (TL)", "Pazar payı (%)", "Yatırımcı sayısı",
                        "Yatırımcı değişimi", "Büyüklük değişimi (%)", "Net akış (TL, tahmini)"] if x in tb.columns]
    s["tablolar"].append(("Seçili fonlar", tb[cols], {"Portföy büyüklüğü (TL)": "{:,.0f}", "Pazar payı (%)": "{:.2%}",
                                                       "Yatırımcı sayısı": "{:,.0f}", "Yatırımcı değişimi": "{:+,.0f}",
                                                       "Büyüklük değişimi (%)": "{:+.2%}",
                                                       "Net akış (TL, tahmini)": "{:,.0f}"}))
    dn = pd.DataFrame({"Kod": tb.index, "Deger": tb["Portföy büyüklüğü (TL)"].values.astype(float)})
    s["grafikler"].append(("Fon büyüklüğü payı", g["donut_chart"](dn, "Kod", "Deger", th)))
    if "Net akış (TL, tahmini)" in tb.columns:
        pdf = pd.DataFrame({"Fon": tb.index, "Akis": tb["Net akış (TL, tahmini)"].astype(float).values})
        s["grafikler"].append(("Tahmini net akış (TL)", g["bar_pos_neg"](pdf, "Fon", "Akis", "", th, ".2s")))
    return s


def sec_varlik(g, c):
    th = g["th"]
    s = _bos("Varlık dağılımı")
    if not c["tefas"]:
        s["uyari"].append("Bu bölüm TEFAS fonları içindir; metinde fon kodu yazın.")
        return s
    if not g["live"]:
        s["uyari"].append("Canlı TEFAS verisi kapalı; bu bölüm için yan panelde açın.")
        return s
    raw_b = g["load_breakdown"](str(pd.Timestamp.today().date()))
    tbl = g["breakdown_table"](raw_b, c["tefas"])
    if tbl.empty:
        s["uyari"].append("Seçilen fonlar için portföy dağılımı verisi bulunamadı.")
        return s
    for f in tbl.index:
        row = tbl.loc[f].dropna()
        row = row[row > 0]
        if len(row):
            s["maddeler"].append(f"**{f}**: en büyük pay {row.idxmax()} (%{row.max():.0f}).")
    s["grafikler"].append(("Fonlara göre dağılım (%)", g["alloc_bar"](tbl, th)))
    s["tablolar"].append(("Portföy dağılımı (%)", tbl.round(1), {}))
    s["maddeler"].append("Oranlar TEFAS portföy dağılımından gelir; toplamın %100 olması gerekmez.")
    return s


def sec_profil(g, c):
    from profil_panel import FON_MAX, PROFILLER, SEPET_ARALIK
    s = _bos("Yatırımcı profili uyumu")
    pr = st.session_state.get("profil")
    if not pr:
        s["uyari"].append("Profil henüz oluşturulmadı. 'Yatırımcı profili' sekmesindeki soruları yanıtlayın, "
                          "sonra bu bölümü yeniden işaretleyin.")
        return s
    P = c["P"]
    profil, idx = PROFILLER[pr["idx"]], pr["idx"]
    s["maddeler"].append(f"Profiliniz **{profil}**: risk kapasitesi {pr['cap']:.1f}/7, risk toleransı {pr['tol']:.1f}/7, "
                         f"risk seviyesi {pr.get('seviye', '-')}/7. Tek bir fonda önerilen en yüksek TEFAS risk "
                         f"değeri: {FON_MAX[idx]}.")
    rows, asan = {}, []
    for col in P.columns:
        rd = g["risk_degeri"](P[col].dropna())[0]
        durum = "Hesaplanamadı" if pd.isna(rd) else ("Uygun" if rd <= FON_MAX[idx] else "Profil sınırının üzerinde")
        if durum == "Profil sınırının üzerinde":
            asan.append(col)
        rows[col] = {"Tür": "ETF" if col in c["etfs"] else "TEFAS", "Risk değeri (1-7)": rd,
                     "Profil üst sınırı": FON_MAX[idx], "Durum": durum}
    s["tablolar"].append(("Fon bazında uyum", pd.DataFrame(rows).T, {"Risk değeri (1-7)": "{:.0f}", "Profil üst sınırı": "{:.0f}"}))
    s["maddeler"].append("Profil sınırının üzerinde risk değerine sahip: " + ", ".join(asan) + "." if asan
                         else "Seçilen varlıkların tamamı profilinizin üst sınırı içinde.")
    cols = list(P.columns)
    res = g["basket_analysis"](P, {x: 1.0 for x in cols}, None, {}, 0.0, 0.0, g["alpha"])
    if res is not None:
        rv = g["risk_degeri"](res["total"])[0]
        if not pd.isna(rv):
            lo, hi = SEPET_ARALIK[idx]
            nasil = "üzerinde" if rv > hi else ("altında" if rv < lo else "içinde")
            s["maddeler"].append(f"Eşit ağırlıklı sepetin risk değeri {int(rv)}/7; {profil} profilin referans aralığının "
                                 f"({lo}-{hi}) {nasil}.")
    s["maddeler"].append("Profil-risk eşleşmesi panelin referans varsayımıdır; TEFAS veya SPK kuralı değildir.")
    return s


BUILDERS = {"sepet": sec_sepet, "perf": sec_perf, "risk": sec_risk, "kars": sec_kars, "etf": sec_etf,
            "akis": sec_akis, "varlik": sec_varlik, "profil": sec_profil}


# ---------- Çıktı: ekran ve HTML ----------
def _goster(secs):
    for s in secs:
        st.markdown(f"### {s['baslik']}")
        for m in s["maddeler"]:
            st.markdown("- " + m)
        for u in s["uyari"]:
            st.warning(u)
        for t, df, fmt in s["tablolar"]:
            st.markdown(f"**{t}**")
            st.dataframe(df.style.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="-"),
                         use_container_width=True)
        for t, ch in s["grafikler"]:
            st.markdown(f"**{t}**")
            st.altair_chart(ch, use_container_width=True, theme=None)
        st.divider()


def _md(x):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _html_mod.escape(x))


def _html(baslik, secs):
    css = ("body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#111827}"
           "h1{margin-bottom:0}h2{border-bottom:2px solid #1d4ed8;padding-bottom:4px;margin-top:36px}"
           "table{border-collapse:collapse;font-size:13px;margin:8px 0 18px}td,th{border:1px solid #e5e7eb;padding:4px 8px;text-align:right}"
           "th{background:#eff4ff}li{margin:4px 0}.disc{color:#6b7280;font-size:12px;margin-top:30px}.warn{background:#fef3c7;padding:8px;border-radius:6px}")
    out = [f"<!doctype html><html lang='tr'><head><meta charset='utf-8'><title>{_html_mod.escape(baslik)}</title>"
           "<script src='https://cdn.jsdelivr.net/npm/vega@5.25.0'></script>"
           "<script src='https://cdn.jsdelivr.net/npm/vega-lite@5.16.3'></script>"
           "<script src='https://cdn.jsdelivr.net/npm/vega-embed@6.23.0'></script>"
           f"<style>{css}</style></head><body><h1>{_html_mod.escape(baslik)}</h1>"
           f"<p>Oluşturulma: {pd.Timestamp.today():%d.%m.%Y}</p>"]
    n = 0
    for s in secs:
        out.append(f"<h2>{_html_mod.escape(s['baslik'])}</h2>")
        if s["maddeler"]:
            out.append("<ul>" + "".join(f"<li>{_md(m)}</li>" for m in s["maddeler"]) + "</ul>")
        for u in s["uyari"]:
            out.append(f"<p class='warn'>{_html_mod.escape(u)}</p>")
        for t, df, fmt in s["tablolar"]:
            out.append(f"<h4>{_html_mod.escape(t)}</h4>")
            try:
                out.append(df.style.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="-").to_html())
            except Exception:
                out.append(df.to_html())
        for t, ch in s["grafikler"]:
            try:
                spec = json.dumps(ch.to_dict(), default=str)
            except Exception:
                continue
            n += 1
            out.append(f"<h4>{_html_mod.escape(t)}</h4><div id='v{n}'></div>"
                       f"<script>vegaEmbed('#v{n}', {spec}, {{actions:false}});</script>")
    out.append(f"<p class='disc'>{_html_mod.escape(DISC)} Grafikler internet bağlantısı gerektirir.</p></body></html>")
    return "".join(out)


# ---------- Arayüz ----------
def _hepsi(deger):
    for k, _ in MODULLER:
        st.session_state[f"bl_c_{k}"] = deger


def render_bulten(g):
    st.markdown("### Fon analiz bülteni")
    st.caption("**1)** Ne incelemek istediğinizi yazın · **2)** Aşağıdan analizleri işaretleyin · **3)** Bülten otomatik oluşur. "
               "Veriler: TEFAS (fonlar) ve Yahoo Finance (ETF'ler).")
    yazi = st.text_area("Ne incelemek istiyorsunuz?", key="bl_yazi", height=110,
                        placeholder='Örn. "AFA ve TI2 fonlarını SPY ve GLD ETF ile karşılaştırmak istiyorum, 200 bin TL, '
                                    'son 1 yıl". Yazdıktan sonra kutunun dışına tıklayın.',
                        help="Fon kodlarını (AFA, TI2), ETF kodlarını (SPY, QQQ, GLD) ve isterseniz tutarı (200 bin TL) "
                             "ve dönemi (son 6 ay, 2 yıl) yazabilirsiniz. Kod yazmazsanız yan paneldeki fonlar kullanılır.")
    tefas, etfs, tutar_m, gun = _coz(g, yazi or "")
    if not tefas and not etfs:
        tefas = list(g["p"].columns)[:8]
        kaynak = "yan panelden" if tefas else ""
    else:
        kaynak = "metinden"
    bilgi = []
    if tefas:
        bilgi.append("TEFAS: " + ", ".join(tefas))
    if etfs:
        bilgi.append("ETF: " + ", ".join(etfs))
    st.caption(("Algılanan (" + kaynak + "): " + " · ".join(bilgi)) if bilgi else
               "Henüz fon veya ETF algılanmadı. Metne bir fon kodu (örn. AFA) veya ETF (örn. SPY) yazın.")

    st.markdown("**Hangi analizler bültende olsun?**")
    b1, b2 = st.columns([1, 5])
    b1.button("Tümünü işaretle", on_click=_hepsi, args=(True,))
    b2.button("Temizle", on_click=_hepsi, args=(False,))
    secili = []
    cols = st.columns(4)
    for i, (k, ad) in enumerate(MODULLER):
        if cols[i % 4].checkbox(ad, key=f"bl_c_{k}"):
            secili.append(k)
    tutar = st.number_input("Toplam yatırım tutarı (TL)", 1000.0, 1e12, float(tutar_m or 100000.0), 10000.0,
                            help="Portföy sepeti ve yatırımcı karşılaştırma bölümlerinde kullanılır. Metinde tutar "
                                 "yazdıysanız (örn. 200 bin TL) otomatik gelir.")

    if not secili:
        st.info("Bülten oluşturmak için yukarıdan en az bir analizi işaretleyin.")
        return
    if not tefas and not etfs:
        st.info("Bülten için metne fon kodu veya ETF yazın ya da yan panele fon kodu ekleyin.")
        return

    end = pd.Timestamp(g["end"])
    start = end - pd.Timedelta(days=gun) if gun else pd.Timestamp(g["start"])
    with st.spinner("Bülten hazırlanıyor (ilk seferde veri çekimi birkaç dakika sürebilir)..."):
        P, ad, tefas_ok, etf_ok, uyar = _veri(g, tefas, etfs, start, end)
        if P is None:
            for u in uyar:
                st.warning(u)
            st.error("Hiçbir varlık için veri alınamadı.")
            return
        ctx = {"P": P, "ad": ad, "tefas": tefas_ok, "etfs": etf_ok, "tutar": float(tutar)}
        secs = []
        try:
            secs.append(sec_ozet(g, ctx))
        except Exception as e:
            secs.append({**_bos("Bülten özeti"), "uyari": [f"Özet oluşturulamadı: {e}"]})
        for k, ad_k in MODULLER:
            if k in secili:
                try:
                    secs.append(BUILDERS[k](g, ctx))
                except Exception as e:
                    secs.append({**_bos(ad_k), "uyari": [f"Bu bölüm oluşturulamadı: {e}"]})
        html_out = _html("Fon Analiz Bülteni", secs)
    for u in uyar:
        st.warning(u)
    st.divider()
    st.download_button("Bülteni indir (HTML)", html_out.encode("utf-8"), "fon_analiz_bulteni.html", "text/html",
                       type="primary")
    _goster(secs)
    st.caption(DISC)
