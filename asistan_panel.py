"""asistan_panel.py - Komut kutusunun 'asistan gibi' cevap veren kısmı.

Niyetler: sepet (mevcut sepet akışı) | liste (sıralama/sorgu) | fon (tek fon incele) | karsilastir | genel (bilgi sorusu).
Varsayılan yorumlama kural tabanlıdır (anahtar gerekmez). Streamlit secrets içinde ANTHROPIC_API_KEY varsa komut önce
Claude ile yapılandırılır ve cevap metni Claude ile yazılır; hata olursa kural tabanlı akışa düşer.
Sayılar her zaman koddan gelir; yapay zekâya yalnızca hesaplanmış veri verilir ve tavsiye vermemesi istenir.
"""
import html as _h
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from gorsel_panel import _chip, _ciz, _rozet, _spark, _strip, stretch

SINIFLAR = ["Para piyasası", "Borçlanma araçları", "Karma / değişken", "Hisse senedi (TR)", "Altın", "Yabancı varlık / ETF"]
STOP = {"TEFAS", "SPK", "KAP", "BIST", "USD", "TRY", "ETF", "ABD", "MDD", "VAR", "CVAR", "FON", "FONU", "BIR", "VE", "ILE", "ICIN", "TL"}
YARDIM = ["5 fon, maksimum getirili sepet oluştur", "En çok kazandıran 10 hisse fonu", "AFT fonunu incele",
          "AFT ile TCD'yi karşılaştır", "Altın fonlarını büyüklüğe göre sırala", "Risk değeri nedir?"]


# ---------- Yapay zekâ (isteğe bağlı) ----------
def llm_aktif():
    try:
        return bool(st.secrets.get("ANTHROPIC_API_KEY"))
    except Exception:
        return False


def llm(sistem, kullanici, max_tokens=700):
    """Claude'a tek mesaj gönderir; anahtar yoksa veya hata olursa None döner."""
    if not llm_aktif():
        return None
    try:
        import requests
        model = st.secrets.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=45,
                          headers={"x-api-key": st.secrets["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": model, "max_tokens": max_tokens, "system": sistem,
                                "messages": [{"role": "user", "content": kullanici}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
    except Exception as e:
        st.session_state["llm_hata"] = f"{type(e).__name__}: {str(e)[:120]}"
        return None


SISTEM_ANLATIM = ("Sen TEFAS fon ve ETF verisini özetleyen bir asistansın. Yatırım tavsiyesi VERME; 'al', 'sat', 'öneririm', "
                  "'tutun' gibi yönlendirici ifadeler kullanma. Yalnızca sana verilen JSON verisindeki sayıları kullan, veri dışında "
                  "rakam, fon adı veya getiri uydurma. Türkçe, sade ve en fazla 6 cümle yaz. Son cümlede geçmiş performansın "
                  "geleceği garanti etmediğini belirt.")


def _anlat(soru, veri, sablon):
    """Cevap metni: Claude varsa onunla, yoksa şablonla."""
    t = llm(SISTEM_ANLATIM, f"Soru: {soru}\nVeri (JSON): {json.dumps(veri, ensure_ascii=False, default=str)[:6000]}", 500)
    return t or sablon


# ---------- Niyet ----------
def _norm(g, s):
    return re.sub(r"[’'`´]", "", s.translate(g["TR_MAP"]).lower())


def _kodlar(g, yazi):
    """Metindeki kodlar. Büyük harfli 3-5 karakterli sözcükler kod adayıdır (listede yoksa 'bulunamadı' sayılır);
    'aft fonu' gibi küçük harfli sözcükler yalnızca gerçekten listede varsa kod sayılır (sınıf sözcükleri karışmasın).
    Dönüş: (listede bulunan kodlar, listede bulunamayan büyük harfli adaylar)."""
    buyuk = [t for t in re.findall(r"\b[A-Za-z][A-Za-z0-9]{2,4}\b", yazi) if t.isupper() and t not in STOP]
    kucuk = [t.upper() for t in re.findall(r"\b([a-z][a-z0-9]{2,4})\s+(?:fon|fonu|fonunu|etf|etfi)\b", yazi)
             if t.upper() not in STOP]
    buyuk, kucuk = list(dict.fromkeys(buyuk)), list(dict.fromkeys(kucuk))
    if not buyuk and not kucuk:
        return [], []
    from veri_panel import etf_veri, tefas_evren
    gecerli = set(tefas_evren(g)["U"].index) | set(etf_veri(g)["E"]["Kod"])
    bulunan = [k for k in dict.fromkeys(buyuk + kucuk) if k in gecerli]
    return bulunan, [k for k in buyuk if k not in gecerli]


def niyet(g, yazi, anla):
    ln = _llm_niyet(g, yazi)
    if ln:
        return ln
    qn = _norm(g, yazi)
    sepet = bool(re.search(r"sepet|portfoy|portfolyo|olustur|hazirla|oner|fon ver|bana .*ver\b", qn))
    liste = bool(re.search(r"en cok|en iyi|en yuksek|en dusuk|en az|en buyuk|en kotu|listele|sirala|hangi|kazandiran|kaybettiren|\btop \d+|ilk \d+", qn))
    genel = bool(re.search(r"nedir|ne demek|nasil|neden|acikla|anlat|\?", qn))
    kodlar, yok = ([], []) if sepet else _kodlar(g, yazi)
    if yok and not kodlar and re.search(r"incele|analiz|hakkinda|karsilastir|kiyasla|nasil|durumu|performans|fonu\b", qn):
        return {"tur": "bulunamadi", "kodlar": yok}
    if len(kodlar) >= 2:
        return {"tur": "karsilastir", "kodlar": kodlar[:5]}
    if len(kodlar) == 1:
        return {"tur": "fon", "kodlar": kodlar}
    if sepet:
        return {"tur": "sepet"}
    if liste:
        return {"tur": "liste", **_liste_spec(qn, anla)}
    if genel:
        return {"tur": "genel"}
    return {"tur": "sepet"}


def _liste_spec(qn, anla):
    n = anla.get("n")
    m = re.search(r"(?:ilk|top|en iyi|en cok kazandiran|en yuksek)\s*(\d{1,2})\b", qn)
    if not n and m:
        n = int(m.group(1))
    metrik, yon = "Getiri1Y", "azalan"
    if re.search(r"buyukl|en buyuk|varlik buyu", qn):
        metrik = "Büyüklük"
    elif "yatirimci" in qn or "yatirimcisi" in qn:
        metrik = "Yatırımcı"
    elif re.search(r"volatilite|dusuk risk|az riskli|riski dusuk|guvenli", qn):
        metrik, yon = "Vol", "artan"
    elif "sharpe" in qn:
        metrik = "Sharpe"
    if re.search(r"en dusuk|en az|en kotu|kaybettiren", qn) and metrik != "Vol":
        yon = "artan"
    etf = bool(re.search(r"etf|yahoo|yabanci|abd|nasdaq|s&p|amerika|altin", qn)) or any(
        s in anla.get("siniflar", []) for s in ("Yabancı varlık / ETF", "Altın"))
    return {"n": int(min(max(n or 10, 3), 25)), "metrik": metrik, "yon": yon, "siniflar": anla.get("siniflar", []), "etf": etf}


def _llm_niyet(g, yazi):
    if not llm_aktif():
        return None
    sistem = ("Kullanıcının Türkçe komutunu JSON'a çevir. SADECE JSON döndür. Alanlar: tur (sepet|liste|fon|karsilastir|genel), "
              "n (tamsayı|null: istenen fon sayısı), hedef (getiri|risk|denge|null), siniflar (şu listeden alt küme: "
              + json.dumps(SINIFLAR, ensure_ascii=False) + "), kodlar (büyük harfli fon/ETF kodları, 3-5 harf; yoksa []), "
              "metrik (getiri|buyukluk|yatirimci|volatilite|sharpe|null), yon (azalan|artan), etf_dahil (true|false), "
              "tutar (TL sayı|null). sepet=fon sepeti/portföy kur; liste=sıralama veya sorgu (en çok kazandıran vb.); "
              "fon=tek fon inceleme; karsilastir=2+ fon karşılaştırma; genel=bilgi sorusu.")
    t = llm(sistem, yazi, 300)
    if not t:
        return None
    try:
        d = json.loads(re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip())
        tur = d.get("tur")
        if tur not in ("sepet", "liste", "fon", "karsilastir", "genel"):
            return None
        sin = [s for s in (d.get("siniflar") or []) if s in SINIFLAR]
        n = d.get("n") if isinstance(d.get("n"), int) else None
        out = {"tur": tur, "siniflar": sin, "n": n, "mod": {"getiri": "getiri", "risk": "risk"}.get(d.get("hedef"))}
        if tur in ("fon", "karsilastir"):
            from veri_panel import etf_veri, tefas_evren
            gecerli = set(tefas_evren(g)["U"].index) | set(etf_veri(g)["E"]["Kod"])
            kodlar = [k for k in (d.get("kodlar") or []) if isinstance(k, str) and k.upper() in gecerli]
            if not kodlar:
                return None
            out["kodlar"] = [k.upper() for k in kodlar][:5]
            out["tur"] = "karsilastir" if len(out["kodlar"]) >= 2 else "fon"
        if tur == "liste":
            mk = {"getiri": "Getiri1Y", "buyukluk": "Büyüklük", "yatirimci": "Yatırımcı", "volatilite": "Vol", "sharpe": "Sharpe"}
            out.update({"metrik": mk.get(d.get("metrik"), "Getiri1Y"), "yon": "artan" if d.get("yon") == "artan" else "azalan",
                        "etf": bool(d.get("etf_dahil")), "n": int(min(max(n or 10, 3), 25))})
        if tur == "sepet" and isinstance(d.get("tutar"), (int, float)):
            out["tutar"] = float(d["tutar"])
        return out
    except Exception:
        return None


# ---------- Cevaplar ----------
def cevapla(g, ny, yazi, anla):
    st.markdown(f"<div class='ch'>{'Yapay zekâ ile yorumlandı' if llm_aktif() else 'Komut yorumlandı'}: "
                f"{ {'liste': 'liste / sıralama', 'fon': 'tek fon incelemesi', 'karsilastir': 'karşılaştırma', 'genel': 'bilgi sorusu', 'bulunamadi': 'kod araması'}[ny['tur']] }</div>",
                unsafe_allow_html=True)
    try:
        if ny["tur"] == "bulunamadi":
            st.warning("Şu kod(lar) TEFAS fon listesinde veya ETF listesinde bulunamadı: " + ", ".join(ny["kodlar"]) +
                       ". Kodu kontrol edin ya da 'Tüm veriler' sekmesinde arayın.")
        elif ny["tur"] == "liste":
            _liste(g, ny, yazi)
        elif ny["tur"] == "fon":
            _fon(g, ny["kodlar"][0], yazi)
        elif ny["tur"] == "karsilastir":
            _karsilastir(g, ny["kodlar"], yazi)
        else:
            _genel(g, yazi)
    except Exception as e:
        st.error(f"Cevap oluşturulamadı: {type(e).__name__}: {e}")
    if st.session_state.get("llm_hata"):
        st.caption(f"Yapay zekâ yanıt vermedi, kural tabanlı cevap gösterildi ({st.session_state['llm_hata']}).")


def _liste(g, ny, yazi):
    from veri_panel import gecmis_cek, goster_tablo, tum_veri
    th, pc = g["th"], g["pc"]
    with st.spinner("Tüm TEFAS verisi taranıyor..."):
        A = tum_veri(g, etf=ny["etf"])
    f = A.copy()
    if ny["siniflar"]:
        f = f[f["Sınıf"].isin(ny["siniflar"])]
    if not ny["etf"]:
        f = f[f["Tür"] == "TEFAS"]
    metrik, asc = ny["metrik"], ny["yon"] == "artan"
    not_ = ""
    if metrik in ("Vol", "Sharpe"):
        # TEFAS için volatilite geçmiş gerektirir: büyüklüğe göre ilk 8 fonu incele
        aday = list(f[(f["Tür"] == "TEFAS") & f["Vol"].isna()].sort_values("Büyüklük", ascending=False)["Kod"].head(8))
        if aday:
            gecmis_cek(g, aday)
            A = tum_veri(g, etf=ny["etf"])
            f = A[A["Kod"].isin(f["Kod"])]
        f = f.dropna(subset=[metrik])
        not_ = " Volatilite ve Sharpe için TEFAS'ta büyüklüğe göre ilk 8 fonun geçmişi incelendi."
    f = f.dropna(subset=[metrik]).sort_values(metrik, ascending=asc)
    top = f.head(ny["n"])
    if top.empty:
        st.warning("Komutunuza uyan kayıt bulunamadı. Sınıf veya ölçütü değiştirmeyi deneyin.")
        return
    ad = {"Getiri1Y": "1 yıllık getiri", "Büyüklük": "portföy büyüklüğü", "Yatırımcı": "yatırımcı sayısı", "Vol": "volatilite",
          "Sharpe": "Sharpe oranı"}[metrik]
    kapsam = (", ".join(ny["siniflar"]) if ny["siniflar"] else "tüm sınıflar") + (" (TEFAS ve ETF)" if ny["etf"] else " (TEFAS)")
    st.markdown(f"### {ad.capitalize()} sıralaması · {'artan' if asc else 'azalan'} · ilk {len(top)}")
    st.caption(f"Kapsam: {kapsam} · {len(f)} kayıt içinden")
    goster_tablo(top, 38 * (len(top) + 1) + 6)
    if metrik in ("Getiri1Y", "Vol", "Sharpe"):
        col = {"Getiri1Y": "Getiri1Y", "Vol": "Vol", "Sharpe": "Sharpe"}[metrik]
        d = top[["Kod", col]].rename(columns={"Kod": "Fon", col: "Getiri"})
        fmt = ".0%" if metrik != "Sharpe" else ".2f"
        _ciz(lambda: g["bar_pos_neg"](d, "Fon", "Getiri", "", th, fmt))
    ilk = top.iloc[0]
    sablon = (f"**{ad.capitalize()}** ölçütüne göre {kapsam} içinde ilk sırada **{ilk['Kod']}** ({str(ilk['Ad'])[:50]}) yer alıyor"
              + (f"; 1 yıllık getirisi {pc(ilk['Getiri1Y'])}" if pd.notna(ilk["Getiri1Y"]) else "")
              + f". Listedeki ilk {len(top)} kaydın 1 yıllık getiri ortalaması {pc(top['Getiri1Y'].mean())}, "
              f"tüm kapsamın medyanı {pc(f['Getiri1Y'].median())}.{not_} Geçmiş performans gelecekteki getirinin garantisi değildir; "
              "bu liste tavsiye değildir.")
    veri = top[["Kod", "Ad", "Sınıf", "Getiri1Y", "Büyüklük", "Yatırımcı", "Vol", "Risk"]].round(4).to_dict("records")
    st.markdown(_anlat(yazi, {"olcut": ad, "kapsam": kapsam, "kayit_sayisi": int(len(f)),
                              "medyan_getiri": float(f["Getiri1Y"].median()), "ilk_kayitlar": veri}, sablon))


def _seri(g, kodlar):
    """Kodlar için fiyat serileri (TEFAS: geçmiş çekilir, ETF: hazır)."""
    from veri_panel import etf_veri, gecmis_cek, tum_veri
    A = tum_veri(g)
    etf = etf_veri(g)["px"]
    tef = [k for k in kodlar if k not in etf.columns]
    cols, fail = {}, []
    if tef:
        px, _, fail, _ = gecmis_cek(g, tef)
        cols.update({k: px[k] for k in px.columns})
    for k in kodlar:
        if k in etf.columns:
            cols[k] = etf[k].dropna()
    return pd.DataFrame(cols).dropna(how="all"), A, fail


def _fon(g, kod, yazi):
    th, rf, pc = g["th"], g["rf"], g["pc"]
    P, A, fail = _seri(g, [kod])
    if P.empty:
        st.warning(f"{kod} için fiyat geçmişi alınamadı{' (' + ', '.join(fail) + ')' if fail else ''}. Biraz sonra tekrar deneyin.")
        return
    s = P[kod].dropna()
    row = A[A["Kod"] == kod].iloc[0]
    one = pd.DataFrame({kod: s})
    et, dt = g["efficiency_table"](one, rf), g["drawdown_table"](one)
    ret = s.iloc[-1] / s.iloc[0] - 1
    vol, sh = et.loc[kod, "Yıllık volatilite (σ)"], et.loc[kod, "Sharpe oranı"]
    mdd = pd.to_numeric(dt.loc[kod, "Maks. düşüş (MDD)"], errors="coerce")
    rd = g["risk_degeri"](s)[0]
    sub = A[(A["Sınıf"] == row["Sınıf"]) & (A["Tür"] == row["Tür"])].dropna(subset=["Getiri1Y"])
    sira = int((sub["Getiri1Y"] > ret).sum()) + 1
    st.markdown(f"### {kod} · {_h.escape(str(row['Ad']))}")
    st.caption(f"{row['Tür']} · {row['Sınıf']}")
    items = [("1 yıllık getiri", pc(ret), "fiyat bazlı", "up" if ret >= 0 else "down"),
             ("Volatilite", pc(vol), "yıllık", ""), ("En büyük düşüş", pc(mdd), "zirveden", "down"),
             ("Sharpe", "-" if pd.isna(sh) else f"{sh:.2f}", "risksiz orana göre", ""),
             ("Risk değeri", "-" if pd.isna(rd) else f"{int(rd)} / 7", "TEFAS ölçeği", ""),
             ("Sınıf sıralaması", f"{sira} / {len(sub)}", f"{row['Sınıf']} içinde getiri", "")]
    if row["Tür"] == "TEFAS":
        items.append(("Büyüklük", f"{row['Büyüklük'] / 1e6:,.0f} mn TL".replace(",", "."), f"{int(row['Yatırımcı']):,} yatırımcı".replace(",", "."), ""))
    _strip(items)
    st.markdown("##### Fiyat seyri (başlangıç = 100)")
    _ciz(lambda: g["line_chart"](one / one.iloc[0] * 100, th, "", ",.0f", 280))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Zirveden düşüş")
        _ciz(lambda: g["line_chart"](one / one.cummax() - 1, th, "", ".0%", 230))
    with c2:
        st.markdown("##### Aylık getiri")
        mr = g["monthly_returns"](one)
        if len(mr):
            _ciz(lambda: g["monthly_heatmap"](mr, th))
    sablon = (f"**{kod}** son 1 yılda {pc(ret)} getiri sağladı; yıllık volatilite {pc(vol)}, en büyük düşüş {pc(mdd)}. "
              f"{row['Sınıf']} sınıfında getiriye göre {len(sub)} kayıt içinde {sira}. sırada. "
              + ("" if pd.isna(rd) else f"TEFAS risk değeri {int(rd)}/7. ")
              + "Geçmiş performans gelecekteki getirinin garantisi değildir; bu bilgi tavsiye değildir.")
    st.markdown(_anlat(yazi, {"kod": kod, "ad": str(row["Ad"]), "sinif": row["Sınıf"], "getiri_1y": float(ret), "volatilite": float(vol),
                              "maks_dusus": float(mdd), "sharpe": None if pd.isna(sh) else float(sh),
                              "risk_degeri": None if pd.isna(rd) else int(rd), "sinif_sirasi": f"{sira}/{len(sub)}"}, sablon))


def _karsilastir(g, kodlar, yazi):
    th, rf, alpha, pc = g["th"], g["rf"], g["alpha"], g["pc"]
    P, A, fail = _seri(g, kodlar)
    P = P.dropna(axis=1, how="all")
    if P.shape[1] < 2:
        st.warning(f"Karşılaştırma için en az iki fonun fiyatı gerekir{' (alınamayan: ' + ', '.join(fail) + ')' if fail else ''}.")
        return
    st.markdown("### " + " ve ".join(P.columns) + " karşılaştırması")
    st.markdown("##### Fiyat seyri (başlangıç = 100)")
    base = P / P.apply(lambda x: x.dropna().iloc[0]) * 100
    _ciz(lambda: g["line_chart"](base, th, "", ",.0f", 300))
    rt = g["risk_table"](P, P.columns[0], rf, alpha).astype(float)
    rows = []
    for k in P.columns:
        s = P[k].dropna()
        rd = g["risk_degeri"](s)[0]
        ad = A.loc[A["Kod"] == k, "Ad"].iloc[0] if (A["Kod"] == k).any() else k
        rows.append(f"<tr><td><b>{_h.escape(k)}</b></td><td style='text-align:left'>{_h.escape(str(ad)[:40])}</td>"
                    f"<td>{pc(s.iloc[-1] / s.iloc[0] - 1)}</td><td>{pc(rt.loc[k, 'Yıllık volatilite'])}</td>"
                    f"<td>{pc(rt.loc[k, 'Maks. düşüş'])}</td><td>{'-' if pd.isna(rt.loc[k, 'Sharpe']) else format(rt.loc[k, 'Sharpe'], '.2f')}</td>"
                    f"<td>{_rozet(rd)}</td></tr>")
    st.markdown("<div class='tw'><table class='tbl'><tr><th>Kod</th><th>Ad</th><th>1Y getiri</th><th>Volatilite</th><th>Maks. düşüş</th>"
                "<th>Sharpe</th><th>Risk</th></tr>" + "".join(rows) + "</table></div>", unsafe_allow_html=True)
    sc = (rt[["Yıllık volatilite", "Yıllık getiri"]].rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
          .rename_axis("Fon").reset_index())
    st.markdown("##### Risk / getiri konumu")
    _ciz(lambda: g["scatter_chart"](sc, th, ""))
    getiri = (P.iloc[-1] / P.apply(lambda x: x.dropna().iloc[0]) - 1)
    en, ed = getiri.idxmax(), getiri.idxmin()
    sablon = (f"1 yıllık getiride **{en}** ({pc(getiri[en])}) önde, **{ed}** ({pc(getiri[ed])}) geride. "
              f"Volatilite {pc(rt['Yıllık volatilite'].min())} ile {pc(rt['Yıllık volatilite'].max())} arasında değişiyor. "
              "Geçmiş performans gelecekteki getirinin garantisi değildir; bu karşılaştırma tavsiye değildir.")
    st.markdown(_anlat(yazi, {"fonlar": {k: {"getiri": float(getiri[k]), "vol": float(rt.loc[k, "Yıllık volatilite"]),
                                             "mdd": float(rt.loc[k, "Maks. düşüş"])} for k in P.columns}}, sablon))


def _genel(g, yazi):
    t = llm("Sen TEFAS yatırım fonları ve yatırım terimleri konusunda eğitim amaçlı bir asistansın. Yatırım tavsiyesi VERME; "
            "'al/sat/öneririm' deme. Kısa, sade Türkçe ile terimi açıkla (en fazla 8 cümle). Kesin bilmediğin şeyi söyleme; "
            "güncel veri verme. Sonda tavsiye olmadığını belirt.", yazi, 600)
    if t:
        st.markdown(t)
        return
    st.info("Bu bir bilgi sorusu gibi görünüyor. Serbest sorulara cevap verebilmem için yapay zekâ bağlantısı gerekir "
            "(Streamlit secrets içinde ANTHROPIC_API_KEY). Bağlantı yokken şu tür komutları anlıyorum:")
    for y in YARDIM[:5]:
        st.markdown(f"- {y}")
