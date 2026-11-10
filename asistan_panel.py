"""asistan_panel.py - Soru-cevap asistanı.

Kural tabanlı: sorudaki anahtar kelimeleri ve fon kodlarını anlar, cevabı TEFAS verisinden kodla hesaplar.
Yapay zeka modeli / API kullanmaz. app.py'nin globals() sözlüğünü parametre olarak alır.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

DISC = ("Bilgilendirme amaçlıdır; yatırım tavsiyesi değildir. Geçmiş performans gelecekteki getirinin "
        "garantisi değildir. Karar vermeden önce fon izahnamesini ve KAP duyurularını inceleyin.")
STOP = {"VAR", "CVAR", "ETF", "SPK", "TEFAS", "KAP", "BIST", "USD", "EUR", "TRY", "FON", "ILE", "SON", "TL",
        "RISK", "VE", "VEYA", "EN", "BU", "NE", "NASIL", "HANGI", "ICIN", "GETIRI", "ILK", "BANA", "SANA"}
KAT_ANAHTAR = [
    (("para piyasasi", "likit"), "Para piyasası"),
    (("borclanma", "tahvil", "bono"), "Borçlanma araçları"),
    (("hisse",), "Hisse senedi"),
    (("altin", "kiymetli maden"), "Altın ve kıymetli maden"),
    (("yabanci", "eurobond", "doviz", "yurtdisi", "dolar"), "Yabancı / eurobond"),
    (("karma", "degisken", "dengeli"), "Karma / çoklu varlık"),
]
DONEM = [("1 Hafta", ("haftalik", "1 hafta", "son hafta", "bu hafta")),
         ("1 Ay", ("aylik", "1 ay", "son ay", "bir ay")),
         ("3 Ay", ("3 ay", "uc ay", "ceyrek")),
         ("6 Ay", ("6 ay", "alti ay", "yarim yil")),
         ("Yıl başı", ("yilbasi", "yil basi", "ybb", "yil basindan")),
         ("1 Yıl", ("yillik", "1 yil", "son yil", "bir yil", "12 ay"))]
SOZLUK_ANAHTAR = ["sharpe", "sortino", "calmar", "omega", "beta", "alfa", "jensen", "korelasyon", "takip hatasi",
                  "tracking", "bilgi orani", "treynor", "drawdown", "maksimum dusus", "volatilite", "asagi sapma",
                  "ewma", "carpiklik", "basiklik", "jarque", "cvar", "expected shortfall", "cornish", "yakalama",
                  "ulcer", "su alti", "kazanc/kayip", "pozitif gun"]


# ---------- Yardımcılar ----------
def _norm(g, s):
    return re.sub(r"[’'`´]", "", s.translate(g["TR_MAP"]).lower())


def _kodlar(g, soru):
    """Sorudaki fon kodları: panel fonları (büyük/küçük harf fark etmez) ve BÜYÜK harfle yazılmış 3-4 harfli kodlar."""
    known = {c.upper(): c for c in g["p"].columns}
    tr = str.maketrans("İŞĞÜÖÇ", "ISGUOC")
    out = []
    L = "A-Za-zÇĞİÖŞÜçğıöşü"
    for m in re.finditer(rf"(?<![{L}])[{L}]{{3,4}}(?![{L}])", soru):
        tok = m.group(0)
        t = tok.upper().translate(tr)
        if t in known:
            c = known[t]
        elif tok.isupper() and t not in STOP:
            c = t
        else:
            continue
        if c not in out:
            out.append(c)
    return out


@st.cache_data(ttl=21600, show_spinner="TEFAS piyasa verisi çekiliyor (ilk seferde birkaç dakika sürebilir)...")
def load_pazar_getiri(end_str):
    """Tüm fonlar için son fiyat ve dönemsel getiri (TEFAS anlık görüntülerinden)."""
    from pytefas import Crawler
    tf = Crawler()

    def snap(ref):
        for back in range(7):
            d = (ref - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            try:
                df = tf.fetch_many(start=d, kinds=("YAT", "EMK", "BYF"), columns="info")
            except Exception:
                df = None
            if df is not None and len(df):
                return df, d
        return None, None

    cur, cur_d = snap(pd.Timestamp(end_str))
    if cur is None:
        raise RuntimeError("Son 7 günde TEFAS fon verisi bulunamadı")
    cur_ts = pd.Timestamp(cur_d)
    base = cur.drop_duplicates("fund_code").set_index("fund_code")
    out = pd.DataFrame(index=base.index)
    out["Ad"] = base["fund_name"].astype(str)
    out["Fiyat"] = pd.to_numeric(base["price"], errors="coerce")
    out["Büyüklük"] = pd.to_numeric(base["portfolio_size"], errors="coerce")
    out["Yatırımcı"] = pd.to_numeric(base["investor_count"], errors="coerce")
    refs = {"1 Hafta": cur_ts - pd.Timedelta(days=7), "1 Ay": cur_ts - pd.Timedelta(days=30),
            "3 Ay": cur_ts - pd.Timedelta(days=91), "6 Ay": cur_ts - pd.Timedelta(days=182),
            "1 Yıl": cur_ts - pd.Timedelta(days=365),
            "Yıl başı": pd.Timestamp(year=cur_ts.year - 1, month=12, day=31)}
    for ad, ref in refs.items():
        df, _ = snap(ref)
        if df is None:
            out[ad] = np.nan
            continue
        pr = pd.to_numeric(df.drop_duplicates("fund_code").set_index("fund_code")["price"], errors="coerce")
        out[ad] = out["Fiyat"] / pr.reindex(out.index) - 1
    out.attrs["tarih"] = cur_d
    return out


def _pazar(g):
    if not g["live"]:
        st.info("Piyasa taraması için yan panelde 'Canlı TEFAS verisi'ni açın.")
        return None
    try:
        mt = load_pazar_getiri(str(pd.Timestamp.today().date()))
    except Exception as e:
        st.error(f"TEFAS piyasa verisi alınamadı: {e}. requirements.txt dosyasında 'pytefas' olmalı.")
        return None
    mt = mt.copy()
    mt["Kategori"] = [g["fund_category"](x) for x in mt["Ad"]]
    return mt


def _seri(g, code, start=None, end=None):
    """Fon fiyat serisi ve adı. Panelde varsa oradan, yoksa TEFAS'tan çeker."""
    p = g["p"]
    if start is None and code in p.columns:
        return p[code].dropna(), g["titles_all"].get(code, code)
    if not g["live"]:
        raise RuntimeError(f"{code} panelde yok; panel dışı fonlar için yan panelde canlı TEFAS verisini açın.")
    s0 = str(start if start is not None else g["start"])
    e0 = str(end if end is not None else g["end"])
    raw = g["load_tefas"]((code,), s0, e0)
    w = g["to_wide"](raw)["price"]
    if code not in w.columns or w[code].dropna().empty:
        raise RuntimeError(f"{code} için TEFAS verisi bulunamadı.")
    ad = code
    if "title" in raw.columns:
        t = raw.dropna(subset=["title"])
        if len(t):
            ad = str(t["title"].iloc[0])
    return w[code].dropna(), ad


def _ozet(satirlar):
    with st.container(border=True):
        st.markdown("**Özet**")
        for s in satirlar:
            st.markdown("- " + s)


def _pf(x, pc):
    return "-" if x is None or pd.isna(x) else pc(float(x))


# ---------- 1) Fon karnesi ----------
def _karne(g, code):
    pc, tl, th, rf, alpha = g["pc"], g["tl"], g["th"], g["rf"], g["alpha"]
    try:
        s, ad = _seri(g, code)
    except Exception as e:
        st.error(str(e))
        return
    if len(s) < 30:
        st.warning(f"{code} için en az 30 gözlem gerekir; yan panelde tarih aralığını genişletin.")
        return
    df1 = s.to_frame(code)
    rd, rvol = g["risk_degeri"](s)
    rt = g["risk_table"](df1, code, rf, alpha)
    pr = g["period_returns"](df1)
    dt = g["drawdown_table"](df1)
    vt = g["volatility_table"](df1, rf)
    kat = g["fund_category"](ad)
    if rt.empty:
        st.warning("Bu fon için yeterli veri yok.")
        return
    r = rt.loc[code]
    ann, vol, sh, mdd, tot = (float(r["Yıllık getiri"]), float(r["Yıllık volatilite"]), float(r["Sharpe"]),
                              float(r["Maks. düşüş"]), float(r["Dönem getirisi"]))
    guncel = float(dt.loc[code, "Güncel düşüş"]) if len(dt) else np.nan
    dip = dt.loc[code, "Dip tarihi"] if len(dt) else "-"

    b = [f"**{code}** · {ad}. Fon adından tahmini kategori: {kat.lower()}."]
    if pd.isna(rd):
        b.append("TEFAS tarzı risk değeri için en az yaklaşık 6 aylık veri gerekir.")
    else:
        b.append(f"Tahmini **risk değeri {int(rd)}/7** (haftalık getirilerden yıllık volatilite %{rvol:.1f}; "
                 f"TEFAS yöntemi, panelin aralığıyla). Günlük verilerden yıllık volatilite {pc(vol)}.")
    b.append(f"Seçilen aralıkta en büyük düşüş **{pc(mdd)}** (dip: {dip}); "
             + ("şu an zirvede." if pd.isna(guncel) or guncel > -0.001 else f"şu an zirvenin {pc(abs(guncel))} altında."))
    p3, p1y = pr.loc[code].get("3 Ay"), pr.loc[code].get("1 Yıl")
    b.append(f"Aralık getirisi {pc(tot)}, yıllıklandırılmış {pc(ann)}. Son 3 ay: {_pf(p3, pc)}, son 1 yıl: {_pf(p1y, pc)}.")
    b.append(f"Yıllık getiri, yan paneldeki risksiz oranın (%{rf * 100:.0f}) {'üzerinde' if ann > rf else 'altında'}; "
             f"Sharpe {sh:.2f}" + (" (risksiz orandan daha düşük getiri)." if sh < 0 else "."))
    try:
        oran = float(vt.loc[code, "Güncel / tarihsel oynaklık"])
        if oran > 1.3:
            b.append(f"Son 3 aydaki oynaklık tüm döneme göre yüksek ({oran:.1f} kat); fon son dönemde daha dalgalı.")
        elif oran < 0.7:
            b.append(f"Son 3 aydaki oynaklık tüm döneme göre düşük ({oran:.1f} kat); fon son dönemde sakinleşmiş.")
    except Exception:
        pass
    try:
        cur, ref, _, _ = g["load_fund_info"](str(pd.Timestamp.today().date()))
        ft = g["fund_table"](cur, ref)
        if code in ft.index:
            f = ft.loc[code]
            akis = f.get("Net akış (TL, tahmini)", np.nan)
            degis = f.get("Yatırımcı değişimi", np.nan)
            txt = (f"Fon büyüklüğü {f['Portföy büyüklüğü (TL)'] / 1e6:,.0f} mn TL, yatırımcı sayısı "
                   f"{f['Yatırımcı sayısı']:,.0f}")
            if not pd.isna(degis):
                txt += f" (son ~30 günde {degis:+,.0f})"
            if not pd.isna(akis):
                txt += f"; tahmini net akış {akis / 1e6:+,.1f} mn TL"
            b.append(txt + ".")
    except Exception:
        pass
    _ozet(b)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Risk değeri (1-7)", "-" if pd.isna(rd) else f"{int(rd)} / 7",
              help="TEFAS'ın fon risk değeriyle aynı yöntem (haftalık getirilerden yıllık volatilite). "
                   "Panel, yan paneldeki aralığı kullandığı için resmi değerden farklı olabilir.")
    k2.metric("Yıllık getiri", pc(ann), help="Seçilen aralıktaki toplam getirinin yıllığa çevrilmiş hali.")
    k3.metric("Yıllık volatilite", pc(vol), help="Günlük getirilerin yıllıklandırılmış standart sapması.")
    k4.metric("Maks. düşüş", pc(mdd), help="Zirveden dibe en büyük değer kaybı.")
    k5.metric("Sharpe", f"{sh:.2f}", help="Risk başına, risksiz oranın üzerindeki getiri.")
    base = s.iloc[0]
    st.altair_chart(g["line_chart"]((s / base * 100).to_frame(code), th, "Fiyat gelişimi (başlangıç = 100)", ",.0f"),
                    use_container_width=True, theme=None)
    with st.expander("Dönemsel getiriler ve düşüş tablosu"):
        pr2 = pr.drop(columns=["Sıra (3 Ay)"], errors="ignore")
        st.dataframe(pr2.style.format("{:+.2%}", na_rep="-"), use_container_width=True)
        if len(dt):
            st.dataframe(dt, use_container_width=True)


# ---------- 2) Karşılaştırma ----------
def _kiyas(g, codes):
    pc, th, rf, alpha = g["pc"], g["th"], g["rf"], g["alpha"]
    seri, ad = {}, {}
    for c in codes[:5]:
        try:
            seri[c], ad[c] = _seri(g, c)
        except Exception as e:
            st.warning(f"{c}: {e}")
    if len(seri) < 2:
        st.error("Karşılaştırma için en az iki fonun verisi gerekir.")
        return
    dfc = pd.concat(seri, axis=1)
    cols = list(dfc.columns)
    rt = g["risk_table"](dfc, cols[0], rf, alpha).astype(float)
    if rt.empty:
        st.warning("Yeterli veri yok.")
        return
    rt.insert(0, "Risk değeri (1-7)", [g["risk_degeri"](dfc[c])[0] for c in rt.index])
    t = rt[["Risk değeri (1-7)", "Dönem getirisi", "Yıllık getiri", "Yıllık volatilite", "Sharpe", "Maks. düşüş"]]

    b = []
    top = t["Yıllık getiri"].idxmax()
    low = t["Yıllık volatilite"].idxmin()
    shp = t["Sharpe"].idxmax()
    dd = t["Maks. düşüş"].idxmax()
    b.append(f"Yıllık getirisi en yüksek **{top}** ({pc(t.loc[top, 'Yıllık getiri'])}); volatilitesi en düşük **{low}** "
             f"({pc(t.loc[low, 'Yıllık volatilite'])}); en küçük maksimum düşüş **{dd}** ({pc(t.loc[dd, 'Maks. düşüş'])}).")
    if top != low:
        b.append(f"Getiri ile risk aynı fonda değil: **{top}** daha yüksek getiri sağlamış ama volatilitesi "
                 f"{pc(t.loc[top, 'Yıllık volatilite'])}; **{low}** daha sakin seyretmiş "
                 f"(getiri {pc(t.loc[low, 'Yıllık getiri'])}).")
    else:
        b.append(f"**{top}** bu aralıkta hem en yüksek getiriyi hem en düşük volatiliteyi göstermiş.")
    b.append(f"Risk başına getiri (Sharpe) en yüksek **{shp}** ({t.loc[shp, 'Sharpe']:.2f}).")
    rv = {c: t.loc[c, "Risk değeri (1-7)"] for c in t.index if not pd.isna(t.loc[c, "Risk değeri (1-7)"])}
    if rv:
        b.append("Tahmini TEFAS risk değerleri: " + ", ".join(f"{c} {int(v)}/7" for c, v in rv.items()) + ".")
    if len(cols) >= 2:
        kor = g["returns_df"](dfc).corr()
        cc = kor.where(~np.eye(len(kor), dtype=bool)).stack()
        if len(cc):
            a, bb = cc.idxmax()
            v = cc.max()
            b.append(f"En çok benzeyen çift: **{a}** ve **{bb}** (korelasyon {v:.2f}). "
                     + ("Birlikte tutmak çeşitlendirme faydasını azaltır." if v > 0.9 else
                        "Getirileri belirgin biçimde ayrışıyor." if v < 0.5 else ""))
    _ozet(b)
    fm = {c: "{:.2%}" for c in t.columns}
    fm.update({"Risk değeri (1-7)": "{:.0f}", "Dönem getirisi": "{:+.2%}", "Yıllık getiri": "{:+.2%}", "Sharpe": "{:.2f}"})
    st.dataframe(t.style.format(fm, na_rep="-"), use_container_width=True)
    base = dfc.apply(lambda s_: s_.dropna().iloc[0])
    st.altair_chart(g["line_chart"](dfc.ffill() / base * 100, th, "Fiyat gelişimi (başlangıç = 100)", ",.0f"),
                    use_container_width=True, theme=None)
    st.caption("Her fon kendi veri aralığında değerlendirilir; ortak tarih aralığı için yan paneldeki tarihleri kullanın.")


# ---------- 3) Piyasa taraması ----------
def _parse_tara(qn):
    kat = [k for anah, k in KAT_ANAHTAR if any(a in qn for a in anah)]
    donem = "1 Yıl"
    for ad, anah in DONEM:
        if any(a in qn for a in anah):
            donem = ad
            break
    n = 8
    m = re.search(r"(?:ilk|en iyi|top)\s*(\d{1,2})", qn) or re.search(r"(\d{1,2})\s*(?:tane|fon)\b", qn)
    if m:
        n = max(1, min(int(m.group(1)), 20))
    rmax = rmin = None
    m = re.search(r"risk\s*(?:degeri)?\s*(?:en fazla|maksimum|max|en cok)?\s*([1-7])\b", qn)
    if m:
        d = int(m.group(1))
        if re.search(rf"{d}\s*(?:ve|veya)?\s*(?:ust|uzer|yuksek)", qn):
            rmin = d
        else:
            rmax = d
    if rmax is None and "dusuk risk" in qn:
        rmax = 3
    if rmin is None and "yuksek risk" in qn:
        rmin = 5
    sirala = "getiri"
    if re.search(r"en (?:dusuk|az) risk", qn):
        sirala = "risk"
    elif "en buyuk" in qn or "buyukluk" in qn:
        sirala = "buyukluk"
    elif "en cok" in qn and "yatirimci" in qn:
        sirala = "yatirimci"
    return {"kat": kat, "donem": donem, "n": n, "rmax": rmax, "rmin": rmin, "sirala": sirala}


def _tara(g, qn):
    pc, rf = g["pc"], g["rf"]
    prm = _parse_tara(qn)
    mt = _pazar(g)
    if mt is None:
        return
    mt = mt[mt["Büyüklük"] >= 5e6]
    if prm["kat"]:
        mt = mt[mt["Kategori"].isin(prm["kat"])]
    else:
        mt = mt[mt["Kategori"] != "GYF / GSYF"]
    d, n = prm["donem"], prm["n"]
    need_risk = prm["rmax"] is not None or prm["rmin"] is not None or prm["sirala"] == "risk"
    if prm["sirala"] == "buyukluk" or prm["sirala"] == "risk":
        mt = mt.sort_values("Büyüklük", ascending=False)
    elif prm["sirala"] == "yatirimci":
        mt = mt.sort_values("Yatırımcı", ascending=False)
    else:
        mt = mt.dropna(subset=[d]).sort_values(d, ascending=False)
    if mt.empty:
        st.warning("Bu kriterlere uyan fon bulunamadı.")
        return
    K = 20 if need_risk else n
    havuz = mt.head(K).copy()
    notlar = []
    if need_risk:
        end = pd.Timestamp(g["end"])
        start = end - pd.Timedelta(days=400)
        rd, shp = {}, {}
        prog = st.progress(0.0, text="Aday fonların risk değeri hesaplanıyor...")
        for i, c in enumerate(havuz.index):
            try:
                s, _ = _seri(g, c, start, end)
                rd[c] = g["risk_degeri"](s)[0]
                et = g["efficiency_table"](s.to_frame(c), rf)
                shp[c] = float(et.loc[c, "Sharpe oranı"]) if len(et) else np.nan
            except Exception:
                rd[c], shp[c] = np.nan, np.nan
            prog.progress((i + 1) / len(havuz))
        prog.empty()
        havuz["Risk değeri"] = pd.Series(rd)
        havuz["Sharpe"] = pd.Series(shp)
        if prm["rmax"] is not None:
            havuz = havuz[havuz["Risk değeri"] <= prm["rmax"]]
        if prm["rmin"] is not None:
            havuz = havuz[havuz["Risk değeri"] >= prm["rmin"]]
        if prm["sirala"] == "risk":
            havuz = havuz.sort_values(["Risk değeri", d], ascending=[True, False])
        notlar.append(f"Risk filtresi için {K} aday fon kontrol edildi (kategori içinde "
                      + ("büyüklüğe" if prm["sirala"] == "risk" else f"{d} getirisine") + " göre ilk "
                      f"{K}); tüm piyasa taranmadı.")
    havuz = havuz.head(n)
    if havuz.empty:
        st.warning("Kriterlere uyan fon bulunamadı. Risk sınırını gevşetmeyi veya kategori belirtmemeyi deneyin.")
        for x in notlar:
            st.caption(x)
        return

    krit = [", ".join(prm["kat"]) if prm["kat"] else "tüm kategoriler"]
    if prm["rmax"] is not None:
        krit.append(f"risk değeri ≤ {prm['rmax']}")
    if prm["rmin"] is not None:
        krit.append(f"risk değeri ≥ {prm['rmin']}")
    sira_ad = {"getiri": f"{d} getirisine göre", "risk": "risk değerine göre (düşükten yükseğe)",
               "buyukluk": "fon büyüklüğüne göre", "yatirimci": "yatırımcı sayısına göre"}[prm["sirala"]]
    b = [f"Kriterler: {'; '.join(krit)}. Sıralama: {sira_ad}. **{len(havuz)} fon** listelendi.",
         f"Listedeki ilk fon **{havuz.index[0]}** ({str(havuz['Ad'].iloc[0])[:50]}); {d} getirisi "
         f"{_pf(havuz[d].iloc[0], pc)}."]
    if "Risk değeri" in havuz and havuz["Risk değeri"].notna().any():
        b.append("Risk değerleri TEFAS yöntemiyle (haftalık getiriler, son ~1 yıl) hesaplanan tahminlerdir.")
    b.append("Yüksek getiri çoğu zaman yüksek oynaklıkla birlikte gelir; risk değeri, maksimum düşüş ve fon "
             "büyüklüğüne de bakın. Dönem getirisi geçmiştir, devamını göstermez.")
    b += notlar
    _ozet(b)
    out = pd.DataFrame({"Ad": havuz["Ad"].str[:48], "Kategori": havuz["Kategori"], d: havuz[d]})
    if "Risk değeri" in havuz:
        out["Risk değeri (1-7)"] = havuz["Risk değeri"]
        out["Sharpe"] = havuz["Sharpe"]
    out["Büyüklük (mn TL)"] = havuz["Büyüklük"] / 1e6
    out["Yatırımcı"] = havuz["Yatırımcı"]
    fm = {d: "{:+.2%}", "Risk değeri (1-7)": "{:.0f}", "Sharpe": "{:.2f}", "Büyüklük (mn TL)": "{:,.0f}",
          "Yatırımcı": "{:,.0f}"}
    out.index.name = "Kod"
    st.dataframe(out.style.format({k: v for k, v in fm.items() if k in out.columns}, na_rep="-"),
                 use_container_width=True)
    st.caption(f"Veri tarihi: {mt.attrs.get('tarih', '-') if hasattr(mt, 'attrs') else '-'} · Kategoriler fon adından "
               "tahmin edilir; 5 mn TL altındaki fonlar listeye alınmaz.")


# ---------- 4) Portföy kontrolü ----------
def _portfoy(g):
    from profil_panel import PROFILLER, SEPET_ARALIK
    pc, tl, th, alpha = g["pc"], g["tl"], g["th"], g["alpha"]
    p, ss = g["p"], st.session_state
    funds = [c for c in ss.get("bk_pick", []) if c in p.columns] or list(p.columns)
    if len(funds) < 2:
        st.info("Portföy kontrolü için en az iki fon gerekir. Yan panele fon kodlarını ekleyin veya "
                "Portföy sepeti sekmesinde fonları seçin.")
        return
    amounts = {c: float(ss.get(f"bk_amt_{c}", 100000.0)) for c in funds}
    rates = {c: g["auto_stopaj"](g["titles_all"].get(c, ""))[0] for c in funds}
    res = g["basket_analysis"](p, amounts, None, rates, 0.0, 0.0, alpha)
    if res is None:
        st.warning("Fonların ortak tarihlerinde yeterli veri yok.")
        return
    t = res["table"]
    rv, rvol = g["risk_degeri"](res["total"])
    uyar, ok = [], []
    wmax = t["Ağırlık"].idxmax()
    if t.loc[wmax, "Ağırlık"] > 0.5:
        uyar.append(f"**Yoğunlaşma:** Sepetin {pc(t.loc[wmax, 'Ağırlık'])}'i tek fonda (**{wmax}**).")
    for c in t.index:
        rc = t.loc[c, "Risk katkısı"]
        if pd.notna(rc) and rc > t.loc[c, "Ağırlık"] + 0.15:
            uyar.append(f"**Risk payı:** **{c}** tutarın {pc(t.loc[c, 'Ağırlık'])}'ini oluşturuyor ama sepet riskinin "
                        f"{pc(rc)}'ini taşıyor.")
    kor = g["returns_df"](p[funds]).corr()
    cc = kor.where(~np.eye(len(kor), dtype=bool)).stack()
    gor = set()
    for (a, bb), v in cc.items():
        if v > 0.9 and (bb, a) not in gor:
            uyar.append(f"**Benzer fonlar:** {a} ve {bb} neredeyse aynı yönde hareket ediyor (korelasyon {v:.2f}); "
                        "birlikte tutmak çeşitlendirmeyi azaltır.")
            gor.add((a, bb))
    if pd.notna(res["div"]) and res["div"] < 1.1:
        uyar.append(f"**Çeşitlendirme:** Oran {res['div']:.2f}; fonlar birbirini dengelemiyor.")
    else:
        ok.append("Fonlar birbirini kısmen dengeliyor (çeşitlendirme oranı "
                  f"{res['div']:.2f}).")
    pr = ss.get("profil")
    if pr and pd.notna(rv):
        lo, hi = SEPET_ARALIK[pr["idx"]]
        if rv > hi:
            uyar.append(f"**Profil uyumu:** Sepet risk değeri {int(rv)}/7; {PROFILLER[pr['idx']]} profil için "
                        f"referans aralığın ({lo}-{hi}) üzerinde.")
        elif rv < lo:
            ok.append(f"Sepet risk değeri {int(rv)}/7; {PROFILLER[pr['idx']]} profil için aralığın ({lo}-{hi}) altında "
                      "(temkinli).")
        else:
            ok.append(f"Sepet risk değeri {int(rv)}/7; {PROFILLER[pr['idx']]} profil aralığında ({lo}-{hi}).")
    brut = (t["Net kâr/zarar (TL)"] + t["Stopaj (TL)"] + t["Masraf (TL)"]).sum()
    kesinti = t["Stopaj (TL)"].sum() + t["Masraf (TL)"].sum()
    if brut > 0 and kesinti > 0:
        ok.append(f"Seçilen aralıkta stopaj ve masraf toplam {tl(kesinti)}; brüt kârın yaklaşık "
                  f"{pc(kesinti / brut)}'i.")
    kay = "Tutarlar Portföy sepeti sekmesinden alınır; girilmemişse eşit tutar (100.000 TL) varsayılır."
    b = [f"{len(funds)} fonluk sepet analiz edildi (seçilen aralıkta brüt getiri {pc(res['ret'])}, yıllık volatilite "
         f"{pc(res['vol'])}, maksimum düşüş {pc(res['dd'])}).", f"Tahmini sepet risk değeri: **{'-' if pd.isna(rv) else str(int(rv)) + '/7'}**."]
    b += ([f"{len(uyar)} dikkat noktası bulundu (aşağıda)."] if uyar else ["Belirgin bir yoğunlaşma veya benzerlik uyarısı yok."])
    b.append(kay)
    _ozet(b)
    if uyar:
        st.markdown("**Dikkat noktaları**")
        for u in uyar:
            st.warning(u)
    if ok:
        st.markdown("**Olumlu / bilgi notları**")
        for o in ok:
            st.success(o)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sepet risk değeri", "-" if pd.isna(rv) else f"{int(rv)} / 7")
    k2.metric("Yıllık volatilite", pc(res["vol"]))
    k3.metric("Maks. düşüş", pc(res["dd"]))
    k4.metric("Çeşitlendirme oranı", "-" if pd.isna(res["div"]) else f"{res['div']:.2f}")
    dn = t["Ağırlık"].rename("Deger").rename_axis("Fon").reset_index()
    st.altair_chart(g["donut_chart"](dn, "Fon", "Deger", th), use_container_width=True, theme=None)
    fm = {"Tutar (TL)": "{:,.0f}", "Ağırlık": "{:.1%}", "Dönem getirisi": "{:+.2%}", "Risk katkısı": "{:.1%}",
          "Net kâr/zarar (TL)": "{:+,.0f}"}
    cols = [c for c in fm if c in t.columns]
    st.dataframe(t[cols].style.format(fm, na_rep="-"), use_container_width=True)


# ---------- 5) Piyasada ne değişti ----------
def _piyasa(g):
    pc, tl, th = g["pc"], g["tl"], g["th"]
    if not g["live"]:
        st.info("Piyasa özeti için yan panelde 'Canlı TEFAS verisi'ni açın.")
        return
    try:
        cur, ref, cur_d, ref_d = g["load_fund_info"](str(pd.Timestamp.today().date()))
        ft = g["fund_table"](cur, ref)
    except Exception as e:
        st.error(f"TEFAS fon bilgisi alınamadı: {e}")
        return
    ct = g["category_table"](ft)
    f = ft[ft["Portföy büyüklüğü (TL)"] >= 5e6]
    b = [f"TEFAS'ta {len(ft):,} fon, toplam {ft['Portföy büyüklüğü (TL)'].sum() / 1e9:,.0f} mlr TL büyüklük ve "
         f"{ft['Yatırımcı sayısı'].sum():,.0f} yatırımcı var (veri tarihi {cur_d}"
         + (f", karşılaştırma {ref_d})." if ref_d else ").")]
    if "Net akış (TL, tahmini)" in f.columns and f["Net akış (TL, tahmini)"].notna().any():
        tot = ft["Net akış (TL, tahmini)"].sum()
        b.append(f"Karşılaştırma tarihinden bu yana tahmini toplam net akış {tot / 1e6:+,.0f} mn TL.")
        if "Net akış (TL, tahmini)" in ct.columns:
            cin, cout = ct["Net akış (TL, tahmini)"].idxmax(), ct["Net akış (TL, tahmini)"].idxmin()
            b.append(f"En çok para girişi **{cin}** kategorisinde ({ct.loc[cin, 'Net akış (TL, tahmini)'] / 1e6:+,.0f} mn TL), "
                     f"en çok çıkış **{cout}** kategorisinde ({ct.loc[cout, 'Net akış (TL, tahmini)'] / 1e6:+,.0f} mn TL).")
        gir = f["Net akış (TL, tahmini)"].nlargest(5)
        cik = f["Net akış (TL, tahmini)"].nsmallest(5)
        b.append(f"En büyük tahmini giriş: **{gir.index[0]}** ({gir.iloc[0] / 1e6:+,.0f} mn TL); en büyük çıkış: "
                 f"**{cik.index[0]}** ({cik.iloc[0] / 1e6:+,.0f} mn TL).")
    b.append("Net akış, pay sayısındaki değişim × güncel fiyat ile hesaplanan yaklaşık bir tahmindir.")
    _ozet(b)
    if "Net akış (TL, tahmini)" in ct.columns:
        d = ct["Net akış (TL, tahmini)"].rename("Akis").rename_axis("Kategori").reset_index()
        st.altair_chart(g["bar_pos_neg"](d, "Kategori", "Akis", "Kategori bazında tahmini net akış (TL)", th, ".2s", 260),
                        use_container_width=True, theme=None)
    if "Net akış (TL, tahmini)" in f.columns:
        c1, c2 = st.columns(2)
        cols = [c for c in ["Fon adı", "Kategori", "Net akış (TL, tahmini)", "Yatırımcı değişimi"] if c in f.columns]
        fm = {"Net akış (TL, tahmini)": "{:+,.0f}", "Yatırımcı değişimi": "{:+,.0f}"}
        with c1:
            st.markdown("**En çok para girişi**")
            st.dataframe(f.nlargest(8, "Net akış (TL, tahmini)")[cols].style.format(fm, na_rep="-"),
                         use_container_width=True)
        with c2:
            st.markdown("**En çok para çıkışı**")
            st.dataframe(f.nsmallest(8, "Net akış (TL, tahmini)")[cols].style.format(fm, na_rep="-"),
                         use_container_width=True)


# ---------- 6) Sözlük ----------
def _sozluk(g, qn):
    alpha = g["alpha"]
    if re.search(r"\bvar\b|riske maruz|guven duzeyi", qn):
        st.markdown(g["var_text"](alpha))
        return True
    if "risk degeri" in qn or "risk skala" in qn:
        st.markdown(g["RISK_ACIKLAMA"] + g["RISK_BANT_MD"])
        return True
    if "stopaj" in qn:
        st.markdown("**Stopaj:** Fon kazancından satışta kesilen vergidir. Panel, gerçek kişi için hisse senedi yoğun "
                    "fonlarda %0, diğer fonlarda genelde %17,5 varsayar ve bunu **fon adından tahmin eder**. Kesin oran "
                    "için fonun TEFAS sayfasındaki 'Stopaj Oranı' alanına ve izahnameye bakın.")
        return True
    ent = []
    for liste in (g["PERF_GLOSSARY"], g["RISK_GLOSSARY"]):
        for e in liste:
            nn = _norm(g, e[0])
            if any(k in qn and k in nn for k in SOZLUK_ANAHTAR):
                ent.append(e)
    if ent:
        g["show_glossary"](ent[:3])
        return True
    return False


# ---------- Yönlendirme ----------
ORNEK = ['"AFA fonunu anlat"', '"AFA ile TI2 kıyasla"', '"Risk değeri 3 ve altında 3 aylık en yüksek getirili para piyasası fonları"',
         '"Portföyümü kontrol et"', '"Piyasada ne değişti?"', '"Sharpe nedir?"']


def _cevapla(g, soru):
    qn = _norm(g, soru)
    kodlar = _kodlar(g, soru)
    if re.search(r"\b(nedir|ne demek|nasil hesaplan|anlami)\b", qn) and not kodlar and _sozluk(g, qn):
        return
    if "portfoy" in qn and re.search(r"kontrol|analiz|saglik|incele|nasil", qn) or "sepetimi" in qn:
        return _portfoy(g)
    if len(kodlar) >= 2:
        return _kiyas(g, kodlar)
    if len(kodlar) == 1:
        return _karne(g, kodlar[0])
    if re.search(r"bugun|ne degisti|piyasada|para giris|para cikis|akis", qn) and not any(
            a in qn for anah, _ in KAT_ANAHTAR for a in anah):
        return _piyasa(g)
    if (any(a in qn for anah, _ in KAT_ANAHTAR for a in anah) or
            re.search(r"en (yuksek|iyi|dusuk|buyuk|cok|az)|risk|getiri|fonlar", qn)):
        return _tara(g, qn)
    if _sozluk(g, qn):
        return
    st.info("Bu soruyu anlayamadım. Şunlardan birini deneyebilirsiniz:\n\n" + "\n".join("- " + x for x in ORNEK))


# ---------- Arayüz ----------
def _soru_ata(soru):
    st.session_state["as_last"] = soru
    st.session_state["as_soru"] = soru
    st.session_state["as_mode"] = None


def _mod_ata(mod):
    st.session_state["as_mode"] = mod
    st.session_state["as_last"] = None


def render_asistan(g):
    p = g["p"]
    st.markdown("### Ne öğrenmek istiyorsun?")
    with st.form("as_form"):
        soru = st.text_input("Sorunuz", key="as_soru", label_visibility="collapsed",
                             placeholder='Örn. "AFA fonunu anlat", "AFA ile TI2 kıyasla", '
                                         '"risk değeri 3 ve altında en yüksek getirili fonlar"')
        ok = st.form_submit_button("Sor", type="primary")
    if ok and soru.strip():
        st.session_state["as_last"] = soru.strip()
        st.session_state["as_mode"] = None

    c1, c2, c3, c4 = st.columns(4)
    c1.button("Portföyümü kontrol et", key="as_c1", on_click=_soru_ata, args=("Portföyümü kontrol et",),
              use_container_width=True)
    c2.button("Bu fonu anlat", key="as_c2", on_click=_mod_ata, args=("fon",), use_container_width=True)
    c3.button("Fonları kıyasla", key="as_c3", on_click=_mod_ata, args=("kiyas",), use_container_width=True)
    c4.button("Piyasada ne değişti?", key="as_c4", on_click=_soru_ata, args=("Piyasada ne değişti?",),
              use_container_width=True)

    mod = st.session_state.get("as_mode")
    if mod == "fon":
        with st.form("as_fon"):
            sec = st.selectbox("Panelden seç", ["—"] + list(p.columns))
            kod = st.text_input("veya fon kodu yaz", placeholder="Örn. AFA")
            if st.form_submit_button("Anlat", type="primary"):
                k = (kod.strip() or (sec if sec != "—" else "")).upper()
                if k:
                    _soru_ata(f"{k} fonunu anlat")
                    st.rerun()
    elif mod == "kiyas":
        with st.form("as_kiyas"):
            kodlar = st.text_input("Fon kodları (virgülle, 2-5 adet)", placeholder="Örn. AFA, TI2",
                                   value=", ".join(list(p.columns)[:2]))
            if st.form_submit_button("Kıyasla", type="primary"):
                ks = [k.strip().upper() for k in kodlar.split(",") if k.strip()]
                if len(ks) >= 2:
                    _soru_ata(f"{' ile '.join(ks[:5])} kıyasla")
                    st.rerun()
                else:
                    st.warning("En az iki fon kodu girin.")

    last = st.session_state.get("as_last")
    if last:
        st.divider()
        st.caption(f"Soru: {last}")
        _cevapla(g, last)
        st.caption(DISC)
    else:
        st.caption("Asistan kural tabanlıdır: sorunuzdaki anahtar kelimeleri ve fon kodlarını anlar, cevabı TEFAS "
                   "verisinden hesaplar (yapay zeka modeli kullanmaz). Örnekler: " + " · ".join(ORNEK[:4]))
