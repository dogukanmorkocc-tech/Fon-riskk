"""bulten_panel.py - Profile uygun fon sepeti akışı (yazı).

Akış: kullanıcı ne incelemek istediğini yazar ve Enter'a basar -> aşağıdan Portföy sepeti / Performans /
Risk / Yatırımcı karşılaştırma kutucuklarını işaretler -> işaretlenenler için yazılı sepet akışı çıkar.
Fon kodu yazmak gerekmez: sepet, TEFAS ve ETF evreninden yatırımcı profiline göre kurulur.
app.py'nin globals() sözlüğünü parametre olarak alır.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

DISC = ("Bu akış kural tabanlı, genel bilgilendirme amaçlı bir örnektir; yatırım tavsiyesi değildir. "
        "Veriler geçmişe dayanır, geçmiş performans gelecekteki getirinin garantisi değildir. Yatırım "
        "danışmanlığı SPK'dan yetkili kuruluşlarca sunulur. Karar vermeden önce fon izahnamesini ve KAP "
        "duyurularını inceleyin.")
MODULLER = [("sepet", "Portföy sepeti"), ("perf", "Performans"), ("risk", "Risk"),
            ("kars", "Yatırımcı karşılaştırma")]
N_PICK = 2          # sınıf başına en fazla fon


# ---------- Metni anlama ----------
def _sayi(sn):
    s = sn.strip()
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        return float(s.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        return float(s.replace(",", ""))
    return float(s.replace(",", "."))


# ---------- Komutu anlama ----------
SAYI_KELIME = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10}
SINIF_ANAHTAR = [  # (düzenli ifade, profil_panel sınıf adı)
    (r"para piyasa|likit", "Para piyasası"),
    (r"borclanma|tahvil|bono|sabit getirili", "Borçlanma araçları"),
    (r"karma|degisken|cok varlikli", "Karma / değişken"),
    (r"turk hisse|bist|hisse senedi|hisse", "Hisse senedi (TR)"),
    (r"altin|kiymetli maden|gumus", "Altın"),
    (r"yabanci|eurobond|dolar|etf|abd|nasdaq|s&p|amerika|global|yurtdisi", "Yabancı varlık / ETF"),
]
MAX_FON = 10          # komutla istenebilecek en yüksek fon sayısı
ONERILEN_MAX = 7      # genelde yeterli sayı; bunun üstünde uyarı verilir


def _anla(g, yazi):
    """Komuttan: fon sayısı, hedef (maksimum getiri / düşük risk), sınıf isteği ve tutarı çıkarır."""
    qn = yazi.translate(g["TR_MAP"]).lower()
    qn = re.sub(r"[’'`´]", "", qn)
    tutar = None
    m = re.search(r"(\d+(?:[.,]\d+)*)\s*(bin|milyon|mn|k)?\s*(?:tl|lira|₺)", qn)
    if m:
        try:
            tutar = _sayi(m.group(1)) * {"bin": 1e3, "k": 1e3, "milyon": 1e6, "mn": 1e6}.get(m.group(2), 1)
        except Exception:
            tutar = None
    # fon sayısı: "5 fon", "yedi tane fon", "10 adet etf"
    n = None
    sk = "|".join(SAYI_KELIME)
    # sayıdan sonra en fazla 4 sözcük (sınıf adı vb.) ve ardından 'fon/etf/varlık': "3 altın ve para piyasası fonu"
    for mm in re.finditer(rf"(?<![\d.,])(\d{{1,2}}|{sk})(?!\d)\s*(?:tane|adet|farkli|ayri|kalem)?\s*"
                          rf"((?:[a-z&]+\s+){{0,4}}?)(?:fon|etf|varlik|urun)", qn):
        t = mm.group(1)
        v = int(t) if t.isdigit() else SAYI_KELIME.get(t)
        ara = mm.group(2)
        if v and 1 <= v <= 20 and not re.search(r"\b(bin|milyon|mn|tl|lira|yil|ay|hafta)\b", ara):
            n = v
            break
    # hedef
    guclu = bool(re.search(r"maksimum getiri|en yuksek getiri|yuksek getiri|en cok kazan|en iyi getiri|getirisi yuksek|"
                           r"getirili|getiri odakli|en cok getiri|gecmiste.*getiri", qn))
    hafif = bool(re.search(r"getiri|kazanc|buyume|agresif", qn))
    dusuk = bool(re.search(r"dusuk risk|guvenli|koruma|temkinli|muhafazakar|az riskli|riski dusuk|dusuk volatil", qn))
    denge = "dengeli" in qn
    mod = "getiri" if guclu else ("risk" if dusuk and not hafif else None)
    siniflar = []
    for rx, ad in SINIF_ANAHTAR:
        if re.search(rx, qn) and ad not in siniflar:
            siniflar.append(ad)
    if "hisse" in qn and re.search(r"abd|nasdaq|s&p|amerika|yabanci|etf", qn) and "Hisse senedi (TR)" in siniflar \
            and not re.search(r"turk|bist|yerli", qn):
        siniflar.remove("Hisse senedi (TR)")          # "ABD hisse" -> yalnızca yabancı varlık / ETF
    odak = []
    if mod == "getiri":
        odak.append("maksimum getiri")
    elif hafif:
        odak.append("getiri")
    if mod == "risk" or dusuk:
        odak.append("düşük risk")
    if denge:
        odak.append("dengeli yapı")
    return {"tutar": tutar, "n": n, "mod": mod, "denge": denge, "siniflar": siniflar, "odak": odak,
            "getiri_w": 1.0 if mod == "getiri" else (0.3 if hafif else 0.0)}


def _kac_fon(komut, idx):
    """İstenen sayı (1-10); yoksa profilin sınıf sayısına göre 5-7."""
    from profil_panel import HEDEF, PROFILLER
    if komut["n"]:
        return int(min(max(komut["n"], 1), MAX_FON))
    k = sum(1 for w in HEDEF[PROFILLER[idx]].values() if w > 0)
    return int(min(max(k, 5), ONERILEN_MAX))


# ---------- Tüm TEFAS evreni (iki tarihte tek seferde) ----------
MIN_BUYUKLUK = 5e7        # 50 mn TL altı fonlar elenir (küçük/likit olmayan)
MIN_YATIRIMCI = 200
DISLA = ("SERBEST", "NITELIKLI", "QUALIFIED", "OZEL FON")


def _evren(g):
    """TEFAS'taki tüm yatırım fonları: bugünkü ve 1 yıl önceki fiyat -> 1 yıllık getiri. Dönüş: (DataFrame, uyarılar, ham sayı)."""
    from profil_panel import KAT_SINIF
    cur, ref, cur_d, ref_d = g["load_fund_info"](str(pd.Timestamp.today().date()), 365)
    num = lambda s: pd.to_numeric(s, errors="coerce")
    c = cur.drop_duplicates("fund_code").set_index("fund_code")
    if "kind" in c.columns and (c["kind"].astype(str) == "YAT").any():
        c = c[c["kind"].astype(str) == "YAT"]
    ham = len(c)
    U = pd.DataFrame({"Fon adı": c["fund_name"].astype(str), "Büyüklük": num(c["portfolio_size"]),
                      "Yatırımcı": num(c["investor_count"]), "Fiyat": num(c["price"])})
    uyar = []
    if ref is not None and len(ref):
        r = ref.drop_duplicates("fund_code").set_index("fund_code").reindex(U.index)
        U["Getiri1Y"] = U["Fiyat"] / num(r["price"]) - 1
    else:
        U["Getiri1Y"] = np.nan
        uyar.append("1 yıl öncesinin fiyatı alınamadı; fonlar büyüklüğe göre sıralandı.")
    U["Sınıf"] = [KAT_SINIF.get(g["fund_category"](n)) for n in U["Fon adı"]]
    ad_up = U["Fon adı"].str.translate(g["TR_MAP"]).str.upper()
    ok = U["Sınıf"].notna() & ~ad_up.apply(lambda x: any(k in x for k in DISLA))
    ok &= (U["Büyüklük"] >= MIN_BUYUKLUK) & (U["Yatırımcı"] >= MIN_YATIRIMCI)
    ok &= U["Getiri1Y"].isna() | U["Getiri1Y"].between(-0.9, 20)      # veri hatalarını ele
    return U[ok], uyar, ham


def _havuz_kodlari(U, komut, idx, n):
    """Tüm evrenden, geçmişi çekilecek finalistleri seçer."""
    from profil_panel import HEDEF, PROFILLER
    hedef = {s: w for s, w in HEDEF[PROFILLER[idx]].items() if w > 0}
    if komut["siniflar"]:
        hedef = {s: hedef.get(s) or 1.0 for s in komut["siniflar"]}
    avail = U["Sınıf"].value_counts()
    kapsam = {s: w for s, w in hedef.items() if avail.get(s, 0) > 0}
    if not kapsam:
        return []
    sort_col = "Getiri1Y" if U["Getiri1Y"].notna().any() else "Büyüklük"
    sec = []
    if komut["mod"] == "getiri":
        cap = max(2, int(np.ceil(n / 2)))
        for s in kapsam:
            sec += list(U[U["Sınıf"] == s].sort_values(sort_col, ascending=False).index[:cap + 1])
        sec = sorted(set(sec), key=lambda x: -U.loc[x, sort_col] if pd.notna(U.loc[x, sort_col]) else 9)[:2 * n + 4]
    else:
        cnt = _dagit(n, kapsam, avail)
        for s, k in cnt.items():
            grp = U[U["Sınıf"] == s]
            by_ret = list(grp.sort_values(sort_col, ascending=False).index[:k + 1])
            by_size = list(grp.sort_values("Büyüklük", ascending=False).index[:k + 1])
            sec += list(dict.fromkeys(by_ret + by_size))[:2 * k + 1]
    return list(dict.fromkeys(sec))


def _dagit(n, w, avail):
    """n fonu sınıflara profil paylarına göre dağıtır (her sınıfa en az 1; n < sınıf sayısıysa en büyük paylar)."""
    sinif = sorted(w, key=w.get, reverse=True)
    if n < len(sinif):
        sinif = sinif[:n]
    tot = sum(w[s] for s in sinif)
    cnt = {s: 1 for s in sinif}
    for _ in range(n - len(sinif)):
        cand = [s for s in sinif if cnt[s] < avail.get(s, 0)]
        if not cand:
            break
        s = max(cand, key=lambda x: w[x] / tot * n - cnt[x])
        cnt[s] += 1
    return cnt


# ---------- Aday havuzu ----------
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version

ARALIK = 11.0        # TEFAS dakikada ~6 istek sınırı uyguluyor: iki ağ çağrısı arası en az bu kadar saniye
_SON = {"t": 0.0}


def _surum(paket):
    try:
        return _pkg_version(paket)
    except PackageNotFoundError:
        return "yüklü değil"


@st.cache_data(ttl=21600, show_spinner=False)
def _fiyat_tek(code, s0, e0):
    """Tek fonun 1 yıllık fiyatı (tefas-crawler >= 0.6.0: fon başına tek istek). Yalnızca başarılı sonuç önbelleğe girer."""
    from tefas import Crawler
    son = None
    for _ in range(2):
        bekle = ARALIK - (time.time() - _SON["t"])
        if bekle > 0:
            time.sleep(bekle)
        _SON["t"] = time.time()
        try:
            try:
                df = Crawler().fetch(start=s0, end=e0, name=code, columns=["code", "date", "price", "title"])
            except TypeError:
                df = Crawler().fetch(start=s0, end=e0, name=code)
            if df is not None and len(df):
                df = df.copy()
                if "code" not in df.columns:
                    df["code"] = code
                df["date"] = pd.to_datetime(df["date"])
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                return df
            son = RuntimeError("boş yanıt (hız sınırı olabilir)")
        except Exception as e:
            son = e
    raise RuntimeError(f"{type(son).__name__}: {str(son)[:100]}")


@st.cache_data(ttl=21600, show_spinner=False)
def _etf_cache(tickers, s0, e0):
    from etf_panel import etf_tl_fiyat
    son = None
    for _ in range(2):
        try:
            df = etf_tl_fiyat(tickers, s0, e0)
            if df is not None and df.shape[1]:
                return df
            son = RuntimeError("boş yanıt")
        except Exception as e:
            son = e
        time.sleep(6)
    raise RuntimeError(str(son)[:120])


def _havuz(g, start, end, komut, pr, n):
    """Tüm TEFAS evrenini tarar, finalistlerin geçmişini çeker; tüm ETF evrenini ekler."""
    from etf_panel import ETF_EVREN
    from profil_panel import ETF_SINIF
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()   # TEFAS düz tarih (YYYY-MM-DD) ister
    s0, e0 = str(start), str(end)
    key = (s0, e0, pr["idx"], n, komut["mod"], tuple(komut["siniflar"]))
    hv = st.session_state.get("akis_havuz")
    if hv and hv["key"] == key:
        return hv
    uyar, cols, fail, hata = [], {}, [], None

    U, ev_uyar, ham = _evren(g)
    uyar += ev_uyar
    kodlar = _havuz_kodlari(U, komut, pr["idx"], n)
    prog = st.progress(0.0, text=f"TEFAS'ta {ham} fon tarandı, {len(U)} uygun fon içinden {len(kodlar)} finalistin geçmişi çekiliyor "
                                 "(dakikada ~6 istek sınırı nedeniyle ilk seferde birkaç dakika sürebilir)...")
    _SON["t"] = time.time()
    iptal = False
    for i, code in enumerate(kodlar, 1):
        if iptal:
            break
        try:
            df = _fiyat_tek(code, s0, e0)
            s = g["to_wide"](df)["price"].get(code)
            if s is None or s.dropna().empty:
                raise ValueError("boş fiyat serisi")
            cols[code] = s
        except Exception as e:
            fail.append(code)
            hata = hata or str(e)
            if not cols and len(fail) >= 3:           # hiç veri gelmiyorsa dakikalarca beklemeyelim
                iptal = True
        prog.progress(i / max(len(kodlar), 1))
    prog.empty()
    cand = pd.DataFrame(cols)
    titles = {c: U.loc[c, "Fon adı"] for c in cand.columns}
    sinif = {c: U.loc[c, "Sınıf"] for c in cand.columns}
    if fail:
        uyar.append("Verisi alınamayan TEFAS fonları: " + ", ".join(fail[:12]) + (" ..." if len(fail) > 12 else "")
                    + (f" · ilk hata: {hata}" if hata else ""))
    if cand.empty:
        uyar.append(f"Kurulu paket sürümleri: tefas-crawler {_surum('tefas-crawler')} "
                    "(fiyat çekimi için 0.6.0 veya üstü gerekir), pytefas " + _surum("pytefas") + ".")

    # ETF'ler (Yahoo Finance): tüm evren tek seferde
    etf_c = {t: ETF_SINIF[k] for t, (_, k) in ETF_EVREN.items() if k in ETF_SINIF}
    etf_c.update({"GLD": "Altın", "IAU": "Altın"})
    if komut["siniflar"]:
        etf_c = {t: s for t, s in etf_c.items() if s in komut["siniflar"]}
    etf_ok = True
    if etf_c:
        try:
            ep = _etf_cache(tuple(etf_c), start, end)
            cand = pd.concat([cand, ep], axis=1).sort_index()
            sinif.update({t: etf_c[t] for t in ep.columns})
        except Exception as e:
            etf_ok = False
            uyar.append(f"ETF adayları alınamadı, yalnızca TEFAS fonları kullanıldı ({e}). "
                        "ETF veri kaynağı geçici olarak yanıt vermiyor olabilir; birkaç dakika sonra deneyin.")
    cand = cand.loc[:, [c for c in cand.columns if c in sinif]]
    if cand.shape[1]:
        nobs = cand.notna().sum()
        cand = cand.loc[:, nobs[nobs >= 60].index]
    hv = {"key": key, "cand": cand, "titles": titles, "sinif": sinif, "uyar": uyar, "etf": set(ETF_EVREN),
          "evren_ham": ham, "evren_uygun": len(U), "finalist": len(kodlar)}
    if cand.shape[1] and not fail and etf_ok:
        st.session_state["akis_havuz"] = hv
    return hv


# ---------- Seçim motoru ----------
def _sec_komut(m, idx, n, mod, siniflar, getiri_w=0.0):
    """m: index=kod; sütunlar Sinif, Sharpe, Vol, MDD, Getiri. Komuta göre n fon ve ağırlıkları seçer."""
    from profil_panel import HEDEF, PROFILLER, SKOR_AGIRLIK
    prof = PROFILLER[idx]
    hedef = {s: w for s, w in HEDEF[prof].items() if w > 0}
    if siniflar:
        hedef = {s: hedef.get(s) or 1.0 for s in siniflar}
    avail = m["Sinif"].value_counts()
    kapsam = {s: w for s, w in hedef.items() if avail.get(s, 0) > 0}
    eksik = [s for s in hedef if s not in kapsam]
    if not kapsam or m.empty:
        return pd.DataFrame(), eksik
    n = int(max(1, min(n, len(m))))
    rows = []
    if mod == "getiri":
        cap = max(2, int(np.ceil(n / 2)))
        sayac = {}
        pool = m[m["Sinif"].isin(kapsam)]
        rank = pool["Getiri"].rank(ascending=False)
        for code, r in pool.sort_values("Getiri", ascending=False).iterrows():
            if sayac.get(r["Sinif"], 0) >= cap:
                continue
            sayac[r["Sinif"]] = sayac.get(r["Sinif"], 0) + 1
            rows.append({"Kod": code, "Sınıf": r["Sinif"], "Ağırlık": 1.0,
                         "Gerekçe": f"{len(pool)} aday içinde getiri sırası {int(rank[code])}"})
            if len(rows) >= n:
                break
        for r in rows:
            r["Ağırlık"] = 1.0 / len(rows)
        return pd.DataFrame(rows), eksik
    cnt = _dagit(n, kapsam, avail)
    tot = sum(kapsam[s] for s in cnt)
    ws, wv, wd = SKOR_AGIRLIK["Muhafazakâr" if mod == "risk" else prof]
    for s, k in cnt.items():
        grp = m[m["Sinif"] == s]
        sc = (ws * grp["Sharpe"].rank(pct=True) + wv * (-grp["Vol"]).rank(pct=True)
              + wd * grp["MDD"].rank(pct=True))
        if getiri_w > 0:
            sc = (1 - getiri_w) * sc + getiri_w * grp["Getiri"].rank(pct=True)
        top = sc.sort_values(ascending=False).head(k).index
        for code in top:
            rows.append({"Kod": code, "Sınıf": s, "Ağırlık": kapsam[s] / tot / len(top),
                         "Gerekçe": f"{len(grp)} aday içinde puan sırası {int(sc.rank(ascending=False)[code])}"})
    d = pd.DataFrame(rows)
    if len(d):
        d["Ağırlık"] = d["Ağırlık"] / d["Ağırlık"].sum()
    return d, eksik


def _sepet_kur(g, hv, pr, tutar, komut, n):
    """Aday havuzundan komuta uygun sepeti seçer. Dönüş: sözlük (bos, sec, eksik, elenen, n, m_all) veya None."""
    from profil_panel import FON_MAX
    cand, sinif = hv["cand"], hv["sinif"]
    if cand.shape[1] == 0:
        return None
    et = g["efficiency_table"](cand, g["rf"])
    dt = g["drawdown_table"](cand)
    m = pd.DataFrame({
        "Sinif": pd.Series(sinif), "Sharpe": et["Sharpe oranı"],
        "Vol": et["Yıllık volatilite (σ)"], "Getiri": et["Yıllıklandırılmış getiri (CAGR)"],
        "MDD": pd.to_numeric(dt["Maks. düşüş (MDD)"], errors="coerce")})
    m = m.loc[m.index.isin(cand.columns)].dropna(subset=["Sinif", "Sharpe", "Vol", "MDD", "Getiri"])
    m["Risk"] = [g["risk_degeri"](cand[c])[0] for c in m.index]
    ok = m["Risk"] <= FON_MAX[pr["idx"]]
    elenen = list(m.index[~ok])
    m_ok = m[ok]
    if m_ok.empty:
        return {"bos": True, "elenen": elenen}
    sec, eksik = _sec_komut(m_ok, pr["idx"], n, komut["mod"], komut["siniflar"], komut["getiri_w"])
    if sec.empty:
        return {"bos": True, "elenen": elenen}
    sec["Tutar"] = sec["Ağırlık"] * tutar
    return {"bos": False, "sec": sec, "eksik": eksik, "elenen": elenen, "n": len(m_ok), "m_all": m}


def _alt_sepetler(g, hv, k, pr, tutar, komut, n, ana_kodlar):
    """Önerilen sepetteki fonlar HARİÇ, aynı havuzdan 2-3 örnek sepet kurar ve analiz eder."""
    from profil_panel import FON_MAX, PROFILLER
    m = k["m_all"].drop(index=[x for x in ana_kodlar if x in k["m_all"].index])
    plan = [("Alternatif 1", "Aynı profil, farklı fonlar", pr["idx"], komut["mod"], False),
            ("Alternatif 2", "Daha temkinli örnek", pr["idx"] - 1, "risk", False),
            ("Alternatif 3", "Daha atak örnek", pr["idx"] + 1, None, True)]
    out = []
    for ad, aciklama, idx, mod, atak in plan:
        if idx < 0 or idx > 4:
            continue
        mm = m[m["Risk"] <= FON_MAX[idx]]
        if len(mm) < 2:
            continue
        sec2, _ = _sec_komut(mm, idx, n, mod, komut["siniflar"], komut["getiri_w"] if ad == "Alternatif 1" else 0.0)
        if sec2.empty or len(sec2) < 2:
            continue
        sec2["Tutar"] = sec2["Ağırlık"] * tutar
        amounts = {x: float(a) for x, a in zip(sec2["Kod"], sec2["Tutar"])}
        rates = {x: (0.0 if x in hv["etf"] else g["auto_stopaj"](hv["titles"].get(x, ""))[0]) for x in amounts}
        res2 = g["basket_analysis"](hv["cand"], amounts, None, rates, 0.0, 0.0, g["alpha"])
        if res2 is None:
            continue
        out.append({"ad": ad, "aciklama": aciklama + f" ({PROFILLER[idx]})", "sec": sec2, "res": res2, "atak": atak})
    return out


# ---------- Yazı bölümleri ----------
def _ad(hv, c):
    t = hv["titles"].get(c)
    if t:
        return str(t).strip()[:45]
    from etf_panel import ETF_EVREN
    return ETF_EVREN.get(c, (c,))[0]


def _pick(paras, prefix):
    return [p for p in paras if p.startswith(prefix)]


def yaz_sepet(g, c):
    from profil_panel import FON_MAX, PROFILLER
    tl, pc = g["tl"], g["pc"]
    sec, pr, hv, k = c["sec"], c["pr"], c["hv"], c["k"]
    profil = PROFILLER[pr["idx"]]
    out = []
    sinif_pay = sec.groupby("Sınıf")["Ağırlık"].sum().sort_values(ascending=False)
    parts = [f"{s} {pc(w)} ({tl(w * c['tutar'])})" for s, w in sinif_pay.items()]
    dag = ("profilinizin örnek varlık dağılımına göre" if c["anla"]["mod"] != "getiri"
           else "maksimum getiri komutunuz nedeniyle eşit ağırlıkla")
    out.append(f"**Sepetin kuruluşu:** {tl(c['tutar'])}, **{profil}** profili için {dag} "
               f"{len(sinif_pay)} sınıfa bölündü: " + "; ".join(parts) + ".")
    for s in sinif_pay.index:
        grp = sec[sec["Sınıf"] == s]
        fonlar = ", ".join(f"**{r['Kod']}** ({_ad(hv, r['Kod'])}; {pc(r['Ağırlık'])}, {tl(r['Tutar'])})"
                           for _, r in grp.iterrows())
        out.append(f"- **{s}:** {fonlar}")
    tefas = [x for x in sec["Kod"] if x not in hv["etf"]]
    etfs = [x for x in sec["Kod"] if x in hv["etf"]]
    kaynak = f"**Kaynak:** Sepet {len(tefas)} TEFAS fonu" + (f" ve {len(etfs)} ETF" if etfs else "") + "'den oluştu."
    if etfs:
        kaynak += " ETF'ler USD/TRY ile TL'ye çevrilmiştir; getirilerine kur etkisi de yansır."
    out.append(kaynak)
    an = c["anla"]
    yontem = (f"**Seçim yöntemi:** TEFAS'ta {hv['evren_ham']} yatırım fonu tarandı; büyüklük (≥50 mn TL), yatırımcı sayısı "
              f"(≥200) ve fon türü filtresinden sonra {hv['evren_uygun']} fon kaldı. Bunlar 1 yıllık getiriye göre "
              f"sıralandı, {hv['finalist']} finalistin ve tüm ETF evreninin geçmişi çekildi. TEFAS risk değeri "
              f"{FON_MAX[pr['idx']]}'i aşan {len(k['m_all']) - k['n']} aday elendi; kalan {k['n']} aday arasından "
              f"komutunuza göre {len(sec)} fon seçildi")
    if an["mod"] == "getiri":
        yontem += ": maksimum getiri istediğiniz için seçim doğrudan geçmiş getiriye göre yapıldı (bir sınıftan en fazla " \
                  f"{max(2, int(np.ceil(c['n_istenen'] / 2)))} fon)"
    elif an["mod"] == "risk":
        yontem += ": düşük risk istediğiniz için puanlamada düşük volatilite ve düşük düşüş öne çıkarıldı"
    else:
        yontem += ": her sınıfta Sharpe oranı, düşük volatilite ve düşük maksimum düşüşe göre puanlandı"
        if an["getiri_w"] > 0:
            yontem += "; getiri isteğiniz nedeniyle puana getiri sırası da eklendi"
    out.append(yontem + ". Her durumda profilinizin risk sınırı aşılmadı.")
    if c.get("duzenlendi"):
        out.append("**Düzenleme:** Ağırlıkları kendiniz değiştirdiniz; yukarıdaki dağılım sizin tablonuza göredir. "
                   "'Seçim yöntemi' paragrafı otomatik öneriyi anlatır.")
    if k["eksik"]:
        out.append("**Not:** " + ", ".join(k["eksik"]) + " sınıfında uygun aday bulunamadığından payları diğer "
                   "sınıflara orantılı dağıtıldı.")
    return out


def yaz_perf(g, c):
    pc = g["pc"]
    res = c["res"]
    out = _pick(c["bt"], "**Getiri:**")
    tot = res["total"]
    parcalar = []
    for ad, off in (("1 ayda", pd.DateOffset(months=1)), ("3 ayda", pd.DateOffset(months=3)),
                    ("6 ayda", pd.DateOffset(months=6))):
        t0 = tot.index[-1] - off
        if t0 >= tot.index[0]:
            v = tot.asof(t0)
            if pd.notna(v) and v > 0:
                parcalar.append(f"son {ad} {pc(tot.iloc[-1] / v - 1)}")
    if parcalar:
        out.append("**Dönemsel seyir:** Sepet değeri " + ", ".join(parcalar) + " değişti (brüt, al-tut).")
    if c["anla"]["getiri_w"] > 0:
        best = res["table"]["Dönem getirisi"]
        out.append(f"**Getiri odağı:** Sepetteki fonlar arasında dönem getirisi en yüksek **{best.idxmax()}** "
                   f"({pc(best.max())}), en düşük **{best.idxmin()}** ({pc(best.min())}) oldu.")
    out += _pick(c["bt"], "**Katkı:**")
    return out


def yaz_risk(g, c):
    from profil_panel import PROFILLER, SEPET_ARALIK
    res, pr = c["res"], c["pr"]
    out = _pick(c["bt"], "**Risk:**")
    rv, rvol = g["risk_degeri"](res["total"])
    if not pd.isna(rv):
        lo, hi = SEPET_ARALIK[pr["idx"]]
        nasil = "üzerinde" if rv > hi else ("altında" if rv < lo else "içinde")
        out.append(f"**Profil uyumu:** Sepetin tahmini TEFAS risk değeri **{int(rv)}/7** (haftalık getirilerden "
                   f"yıllık volatilite %{str(round(rvol, 1)).replace('.', ',')}); {PROFILLER[pr['idx']]} profilin referans aralığının ({lo}-{hi}) "
                   f"**{nasil}**.")
    out += _pick(c["bt"], "**Çeşitlendirme:**")
    out += _pick(c["bt"], "**Risk yoğunlaşması:**") + _pick(c["bt"], "**Risk dağılımı:**")
    return out


def yaz_tek_fon(g, c):
    tl, pc, rf = g["tl"], g["pc"], g["rf"]
    res, hv = c["res"], c["hv"]
    funds = list(res["table"].index)
    sub = hv["cand"][funds].loc[res["start"]:]
    rates = {x: (0.0 if x in hv["etf"] else g["auto_stopaj"](hv["titles"].get(x, ""))[0]) for x in funds}
    m = g["investor_metrics"](sub, funds, c["tutar"], None, rf, rates, 0.0, 0.0)
    if m.empty:
        return ["Karşılaştırma için yeterli veri yok."]
    t = res["table"]
    tot, net = t["Tutar (TL)"].sum(), t["Net değer (TL)"].sum()
    sepet_net = net / tot - 1
    nets = m["Net getiri"].dropna().astype(float)
    out = []
    if len(nets):
        sira = int((nets > sepet_net).sum()) + 1
        out.append(f"**Sepet ve tek tek fonlar:** Aynı {tl(c['tutar'])} yalnızca tek bir fonda değerlendirilseydi net "
                   f"getiri (stopaj ve masraf sonrası) **{nets.idxmin()}** fonunda {pc(nets.min())}, **{nets.idxmax()}** "
                   f"fonunda {pc(nets.max())} olurdu. Sepetin net getirisi {pc(sepet_net)}; bu, {len(nets)} fon "
                   f"içinde {sira}. sıraya denk geliyor.")
        lv = m["Yıllık volatilite"].dropna().astype(float)
        dd = m["Maks. düşüş"].dropna().astype(float)
        if len(lv) and len(dd):
            out.append(f"**Risk tarafı:** Fonların volatilitesi {pc(lv.min())} – {pc(lv.max())} arasında; sepetin "
                       f"volatilitesi {pc(res['vol'])}. Tek tek fonların maksimum düşüşü {pc(dd.max())} ile {pc(dd.min())} "
                       f"arasında; sepetin maksimum düşüşü {pc(res['dd'])}.")
    gun = max(int(res["days"]), 1)
    rf_getiri = (1 + rf) ** (gun / 365.25) - 1
    fark = sepet_net - rf_getiri
    out.append(f"**Risksiz getiri ile kıyas:** Aynı sürede %{rf * 100:.0f} yıllık risksiz oran {pc(rf_getiri)} "
               f"getiri sağlardı; sepetin net getirisi bunun {pc(abs(fark))} puan "
               f"{'üzerinde' if fark >= 0 else 'altında'} kaldı.")
    out.append("Stopaj oranları TEFAS fonları için fon adından tahmin edilmiştir; ETF için vergi ve komisyon 0 "
               "varsayılmıştır. Kesin oranı izahnameden teyit edin.")
    return out


def yaz_kars(g, c):
    pc = g["pc"]
    alts = c.get("alts", [])
    ana = {"ad": "Önerilen sepet", "aciklama": "sizin fonlarınız", "sec": c["sec"], "res": c["res"], "atak": False}
    out = []
    if not alts:
        out.append("**Örnek sepetler:** Havuzda, önerilen sepetteki fonların dışında karşılaştırma için yeterli fon "
                   "bulunamadı; aşağıda aynı tutarın tek bir fonda değerlendirilmesi karşılaştırılıyor.")
    else:
        out.append(f"**Karşılaştırma mantığı:** Önerilen sepetteki fonlar dışında kalan adaylardan {len(alts)} örnek sepet "
                   "kuruldu. Hepsi aynı tutar ve aynı dönemle hesaplandı; amaç, farklı fonlarla ve farklı risk "
                   "seviyelerinde sonucun nasıl değiştiğini göstermektir.")
        rows = []
        for s in [ana] + alts:
            res, t = s["res"], s["res"]["table"]
            tot, net = t["Tutar (TL)"].sum(), t["Net değer (TL)"].sum()
            rv = g["risk_degeri"](res["total"])[0]
            rows.append((s["ad"], s.get("aciklama", ""), list(s["sec"]["Kod"]), net / tot - 1, res["vol"], res["dd"], rv, s["atak"]))
            out.append(f"- **{s['ad']}** ({s.get('aciklama', '')}): {', '.join(s['sec']['Kod'])}. Net getiri {pc(net / tot - 1)}, "
                       f"volatilite {pc(res['vol'])}, en büyük düşüş {pc(res['dd'])}"
                       + ("" if pd.isna(rv) else f", risk değeri {int(rv)}/7") + ".")
        en_net = max(rows, key=lambda r: r[3])
        en_vol = min(rows, key=lambda r: r[4])
        sira = sorted(rows, key=lambda r: -r[3]).index(rows[0]) + 1
        out.append(f"**Özet:** En yüksek net getiriyi **{en_net[0]}** ({pc(en_net[3])}), en düşük volatiliteyi "
                   f"**{en_vol[0]}** ({pc(en_vol[4])}) sağladı. Önerilen sepet getiriye göre {len(rows)} sepet içinde "
                   f"{sira}. sırada. Daha yüksek getiri genellikle daha yüksek dalgalanma ve düşüşle birlikte geldi; "
                   "sepetleri yalnızca getiriye göre değil, kendi risk toleransınıza göre değerlendirin.")
        if any(r[7] for r in rows):
            out.append("**Not:** 'Daha atak' örnek sepet profilinizin risk düzeyinin üzerinde olabilir; yalnızca "
                       "karşılaştırma amacıyla gösterilmiştir.")
    return out + yaz_tek_fon(g, c)


def _istek_notu(anla, idx, profil):
    """Yazılan istek profille çelişiyorsa kullanıcıyı açıkça bilgilendirir; sepet her zaman profile göre kurulur."""
    o = anla["odak"]
    nt = []
    if "dengeli yapı" in o and idx != 2:
        nt.append(f"'dengeli' ifadesi {profil} profilinizden farklı")
    if "düşük risk" in o and idx >= 3:
        nt.append(f"'düşük risk' isteği {profil} profilinizin risk düzeyinin altında")
    if "maksimum getiri" in o and idx <= 1:
        nt.append(f"'maksimum/yüksek getiri' isteği {profil} profilinizin risk sınırıyla bağdaşmıyor")
    if not nt:
        return ""
    return ("**Not:** " + "; ".join(nt) + ". Sepet güvenlik için her zaman profilinizin risk sınırlarına göre "
            "kuruldu. Farklı bir risk düzeyi istiyorsanız profil anketini gözden geçirin.")


YAZARLAR = {"sepet": yaz_sepet, "perf": yaz_perf, "risk": yaz_risk, "kars": yaz_kars}


# ---------- Düzenlenebilir ağırlık tablosu ----------
def _editor(g, hv, k, pr, tutar):
    from gorsel_panel import stretch
    """Otomatik sepeti düzenlenebilir tabloya döker. Dönüş: (sec_edit, duzenlendi, uyarilar)."""
    from profil_panel import FON_MAX
    sec = k["sec"]
    auto_w = {r["Kod"]: round(float(r["Ağırlık"]) * 100, 1) for _, r in sec.iterrows()}
    sig = f"{pr['idx']}|{int(tutar)}|{hv['key']}|{'-'.join(sec['Kod'])}"
    sw = st.session_state.setdefault("bl_w", {})
    if sw.get("sig") != sig:                       # yeni sepet: kullanıcı düzenlemelerini sıfırla
        sw.clear()
        sw.update({"sig": sig, "w": dict(auto_w), "ver": sw.get("ver", 0)})
    w = sw["w"]

    st.markdown("#### Sepeti düzenleyin")
    st.caption("**Ağırlık %** sütununa çift tıklayıp değiştirin. Toplam %100 olmak zorunda değil, otomatik %100'e "
               "oranlanır. 0 yazarsanız fon sepetten çıkar. Grafikler ve yazı anında güncellenir. "
               "Önce fon ekleyin, sonra ağırlıkları düzenleyin.")
    pool = [x for x in hv["cand"].columns if x not in auto_w]
    ekle = st.multiselect("Havuzdan fon ekle", pool, key=f"bl_ekle_{sw['ver']}",
                          format_func=lambda x: f"{x} · {_ad(hv, x)} ({hv['sinif'].get(x, '')})")
    kodlar = list(auto_w) + list(ekle)
    df = pd.DataFrame([{"Kod": x, "Fon": _ad(hv, x), "Sınıf": hv["sinif"].get(x, ""),
                        "Ağırlık %": float(w.get(x, 0.0))} for x in kodlar])
    de = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor")
    ed = stretch(de, df, hide_index=True, disabled=["Kod", "Fon", "Sınıf"], key=f"bl_ed_{sw['ver']}",
                 column_config={"Ağırlık %": st.column_config.NumberColumn("Ağırlık %", min_value=0.0, max_value=100.0,
                                                                           step=0.5, format="%.1f")})
    for _, r in ed.iterrows():
        w[r["Kod"]] = float(r["Ağırlık %"]) if pd.notna(r["Ağırlık %"]) else 0.0

    def _sifirla():
        sw["ver"] = sw.get("ver", 0) + 1
        sw["sig"] = None
    st.button("Önerilen ağırlıklara dön", key="bl_reset", on_click=_sifirla)

    wd = {x: max(w.get(x, 0.0), 0.0) for x in kodlar}
    top = sum(wd.values())
    uyar = []
    if top <= 0:
        st.warning("Tüm ağırlıklar 0; önerilen sepet kullanılıyor.")
        wd, top = dict(auto_w), sum(auto_w.values())
    st.caption(f"Girilen toplam: %{top:.1f} → %100'e oranlanır.")
    rows = []
    for x in kodlar:
        if wd.get(x, 0) > 0:
            rows.append({"Kod": x, "Sınıf": hv["sinif"].get(x, ""), "Ağırlık": wd[x] / top, "Tutar": wd[x] / top * tutar})
    sec_e = pd.DataFrame(rows)
    for _, r in sec_e.iterrows():
        rd = g["risk_degeri"](hv["cand"][r["Kod"]].dropna())[0]
        if not pd.isna(rd) and rd > FON_MAX[pr["idx"]]:
            uyar.append(f"**{r['Kod']}** fonunun TEFAS risk değeri {int(rd)}; profilinizin fon sınırı {FON_MAX[pr['idx']]}.")
    duz = (set(sec_e["Kod"]) != set(auto_w)) or any(abs(r["Ağırlık"] * 100 - auto_w.get(r["Kod"], 0)) > 0.3
                                                      for _, r in sec_e.iterrows())
    return sec_e, duz, uyar


# ---------- Arayüz ----------
TAB_ADLARI = [("sepet", "Portföy sepeti"), ("perf", "Performans"), ("risk", "Risk"), ("kars", "Yatırımcı karşılaştırma")]


def _komut_ozeti(g, anla, n, pr, tutar):
    from profil_panel import PROFILLER
    parca = [f"<span class='ch'>{n} fon{' (istenen)' if anla['n'] else ' (profile göre)'}</span>"]
    parca.append("<span class='ch'>Hedef: " + {"getiri": "maksimum getiri", "risk": "düşük risk"}.get(anla["mod"], "profil dengesi") + "</span>")
    if anla["siniflar"]:
        parca.append("<span class='ch'>Sınıf: " + ", ".join(anla["siniflar"]) + "</span>")
    parca.append(f"<span class='ch'>Tutar: {g['tl'](tutar)}</span><span class='ch'>Profil: {PROFILLER[pr['idx']]}</span>")
    st.markdown("<div style='margin:6px 0 10px'><b>Komutunuz şöyle anlaşıldı:</b> " + " ".join(parca) + "</div>",
                unsafe_allow_html=True)


def render_bulten(g):
    from gorsel_panel import GORSEL, hero
    from profil_panel import FON_MAX, PROFILLER
    from tema_panel import ONAY_METNI
    pr0 = st.session_state.get("profil")
    if pr0:
        st.success(f"Profiliniz: **{PROFILLER[pr0['idx']]}** · tutar {g['tl'](pr0['tutar'])}. "
                   "Değiştirmek için üstteki 'Yatırımcı profili' sekmesine geçin.")
    else:
        st.info("Başlamadan önce üstteki **Yatırımcı profili** sekmesine tıklayıp profilinizi oluşturun.")
    st.caption("Ne istediğinizi yazın; fon sepeti kurar, fon listeler, tek fonu inceler veya fonları karşılaştırırım. Örnekler: *“5 fon, maksimum getirili sepet”*, "
               "*“3 altın ve para piyasası fonu”*, *“7 fonluk dengeli sepet, 200 bin TL”*, *“düşük riskli 4 fon”*. "
               "Fon sayısı yazmazsanız profilinize göre 5-7 fon seçilir. Dönem her zaman son 1 yıldır.")
    yazi = st.text_input("Ne incelemek istiyorsun?", key="bl_yazi",
                         placeholder='Örn. "5 fonluk maksimum getirili sepet oluştur"',
                         help="Fon sayısı (beş, 7, on fon), hedef (maksimum getiri, düşük risk, dengeli), sınıf "
                              "(altın, hisse, para piyasası, borçlanma, yabancı/ETF) ve tutar (200 bin TL) yazabilirsiniz.")
    if not (yazi or "").strip():
        st.info("Ne incelemek istediğinizi yazın.")
        return
    onay = st.checkbox(ONAY_METNI, key="bl_onay")
    st.button("Oluştur", type="primary", key="bl_btn",
              on_click=lambda: st.session_state.__setitem__("bl_son", (st.session_state.get("bl_yazi") or "").strip()))
    if st.session_state.get("bl_son") != yazi.strip():
        st.info("Yazdıktan sonra 'Oluştur'a basın.")
        return
    if not onay:
        st.warning("Sonucu görmek için yukarıdaki yasal uyarı onay kutusunu işaretleyin.")
        return
    from asistan_panel import cevapla, niyet
    anla = _anla(g, yazi)
    with st.spinner("Komutunuz yorumlanıyor..."):
        ny = niyet(g, yazi, anla)
    if ny["tur"] != "sepet":                      # liste / tek fon / karşılaştırma / soru: profil gerekmez
        cevapla(g, ny, yazi, anla)
        from tema_panel import yasal_alt
        yasal_alt()
        return
    if ny.get("n"):
        anla["n"] = ny["n"]
    if ny.get("mod"):
        anla["mod"] = ny["mod"]
        anla["getiri_w"] = 1.0 if ny["mod"] == "getiri" else anla["getiri_w"]
    if ny.get("siniflar"):
        anla["siniflar"] = ny["siniflar"]
    if ny.get("tutar"):
        anla["tutar"] = ny["tutar"]
    pr = st.session_state.get("profil")
    if not pr:
        st.warning("Sepet profilinize göre kurulur. Önce 'Yatırımcı profili' sekmesinde anketi doldurup "
                   "'Profilimi hesapla'ya basın.")
        return
    if not g["live"]:
        st.warning("Sepet için canlı TEFAS verisi gerekir.")
        return

    tutar = float(anla["tutar"] or pr["tutar"])
    n = _kac_fon(anla, pr["idx"])
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=1)          # dönem her zaman son 1 yıl

    with st.spinner("Komutunuz işleniyor: tüm TEFAS evreni taranıyor..."):
        try:
            hv = _havuz(g, start, end, anla, pr, n)
        except Exception as e:
            st.error(f"Aday fon verisi alınamadı: {type(e).__name__}: {e}")
            return
        k = _sepet_kur(g, hv, pr, tutar, anla, n)
    if k is None:
        st.warning("Aday fon verisi alınamadı. Nedenler:")
        for u in hv["uyar"]:
            st.caption(u)
        return
    if k["bos"]:
        st.warning(f"TEFAS risk değeri {FON_MAX[pr['idx']]} veya altında, komutunuza uyan aday bulunamadı. "
                   "Sınıf isteğinizi genişletmeyi deneyin.")
        return
    for u in hv["uyar"]:
        st.caption(u)

    _komut_ozeti(g, anla, n, pr, tutar)
    if anla["n"] and anla["n"] > ONERILEN_MAX:
        st.info(f"{anla['n']} fon istediniz; sepet bu sayıya göre kuruldu. Not: Bireysel portföyler için genellikle "
                f"{5}-{ONERILEN_MAX} fon yeterlidir; fon sayısı arttıkça çeşitlendirme faydası azalır, takip zorlaşır.")
    if anla["n"] and anla["n"] > MAX_FON:
        st.info(f"En fazla {MAX_FON} fon desteklenir; sepet {MAX_FON} fonla kuruldu.")
    if len(k["sec"]) < n:
        st.info(f"{n} fon hedeflendi; komutunuza ve profilinizin risk sınırına uyan {len(k['sec'])} fon bulundu.")
    uyum = _istek_notu(anla, pr["idx"], PROFILLER[pr["idx"]])
    if uyum:
        st.info(uyum)

    hero_slot = st.container()                    # özet kartı en üstte görünsün; hesap tablodan sonra yapılır
    sekmeler = st.tabs([ad for _, ad in TAB_ADLARI])
    with sekmeler[0]:
        sec, duz, uyarilar = _editor(g, hv, k, pr, tutar)
        for u in uyarilar:
            st.warning(u)

    amounts = {x: float(a) for x, a in zip(sec["Kod"], sec["Tutar"])}
    rates = {x: (0.0 if x in hv["etf"] else g["auto_stopaj"](hv["titles"].get(x, ""))[0]) for x in amounts}
    res = g["basket_analysis"](hv["cand"], amounts, None, rates, 0.0, 0.0, g["alpha"])
    if res is None:
        st.warning("Seçilen varlıkların ortak tarihlerinde yeterli veri yok. Ağırlıkları değiştirmeyi deneyin.")
        return
    try:
        alts = _alt_sepetler(g, hv, k, pr, tutar, anla, n, list(sec["Kod"]))
    except Exception as e:
        alts = []
        st.caption(f"Örnek sepetler oluşturulamadı: {type(e).__name__}: {e}")
    gun = max(int(res["days"]), 1)
    c = {"pr": pr, "hv": hv, "k": k, "sec": sec, "res": res, "tutar": tutar, "anla": anla, "duzenlendi": duz,
         "alts": alts, "n_istenen": n, "bt": g["basket_text"](res, f"seçilen aralık ({gun} gün)")}
    with hero_slot:
        try:
            hero(g, c)
        except Exception as e:
            st.error(f"Özet kartı oluşturulamadı: {type(e).__name__}: {e}")

    for tab, (key, ad) in zip(sekmeler, TAB_ADLARI):
        with tab:
            try:
                GORSEL[key](g, c)
            except Exception as e:
                st.error(f"Bu sekmenin görselleri oluşturulamadı: {type(e).__name__}: {e}")
            try:
                paras = YAZARLAR[key](g, c)
            except Exception as e:
                paras = [f"Bu bölüm oluşturulamadı: {e}"]
            st.markdown("##### Yazılı değerlendirme")
            for p_ in paras:
                st.markdown(p_)
    from tema_panel import yasal_alt
    yasal_alt()
