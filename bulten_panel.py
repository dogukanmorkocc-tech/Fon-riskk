"""bulten_panel.py - Profile uygun fon sepeti akışı (yazı).

Akış: kullanıcı ne incelemek istediğini yazar ve Enter'a basar -> aşağıdan Portföy sepeti / Performans /
Risk / Yatırımcı karşılaştırma kutucuklarını işaretler -> işaretlenenler için yazılı sepet akışı çıkar.
Fon kodu yazmak gerekmez: sepet, TEFAS ve ETF (Yahoo Finance) evreninden yatırımcı profiline göre kurulur.
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


def _anla(g, yazi):
    """Yazıdan tutar, dönem ve odak (getiri vurgusu) çıkarır. Fon adı aranmaz."""
    qn = yazi.translate(g["TR_MAP"]).lower()
    qn = re.sub(r"[’'`´]", "", qn)
    tutar = None
    m = re.search(r"(\d+(?:[.,]\d+)*)\s*(bin|milyon|mn|k)?\s*(?:tl|lira|₺)", qn)
    if m:
        try:
            tutar = _sayi(m.group(1)) * {"bin": 1e3, "k": 1e3, "milyon": 1e6, "mn": 1e6}.get(m.group(2), 1)
        except Exception:
            tutar = None
    gun = None
    for m in re.finditer(r"(?<!\d)(\d{1,3})\s*(yil|ay|hafta)", qn):
        n, birim = int(m.group(1)), m.group(2)
        if 0 < n <= {"yil": 30, "ay": 360, "hafta": 520}[birim]:
            gun = n * {"yil": 365, "ay": 30, "hafta": 7}[birim]
            break
    yuksek = bool(re.search(r"yuksek getiri|yuksek kazanc|getirili|getirisi yuksek|agresif|buyume", qn))
    getiri = yuksek or bool(re.search(r"getiri|kazanc", qn))
    koruma = bool(re.search(r"dusuk risk|guvenli|koruma|temkinli|muhafazakar|az riskli", qn))
    denge = "dengeli" in qn
    odak = []
    if yuksek:
        odak.append("yüksek getiri")
    elif getiri:
        odak.append("getiri")
    if koruma:
        odak.append("düşük risk")
    if denge:
        odak.append("dengeli yapı")
    return {"tutar": tutar, "gun": gun, "odak": odak,
            "getiri_w": 0.5 if yuksek else (0.3 if getiri else 0.0)}


# ---------- Aday havuzu ----------
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version

ARALIK = 11.0        # TEFAS dakikada ~6 istek sınırı uyguluyor: iki ağ çağrısı arası en az bu kadar saniye
N_ADAY = 2           # sınıf başına hedef aday sayısı
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


def _havuz(g, start, end):
    """TEFAS (her kategoride en büyük fonlar) + ETF evreninden aday fiyatları ve sınıfları."""
    from etf_panel import ETF_EVREN
    from profil_panel import ETF_SINIF, KAT_SINIF
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()   # TEFAS düz tarih (YYYY-MM-DD) ister
    s0, e0 = str(start), str(end)
    key = (s0, e0)
    hv = st.session_state.get("akis_havuz")
    if hv and hv["key"] == key:
        return hv
    uyar, cols, titles, fail, hata = [], {}, {}, [], None

    # 1) Fon listesi ve büyüklükler (pytefas)
    cur, ref, _, _ = g["load_fund_info"](str(pd.Timestamp.today().date()))
    ft = g["fund_table"](cur, ref)
    ft = ft[ft["Kategori"].isin(KAT_SINIF)]
    if "Tür" in ft.columns and (ft["Tür"] == "YAT").any():
        ft = ft[ft["Tür"] == "YAT"]                       # emeklilik fonları sepete alınmaz
    _SON["t"] = time.time()

    # 2) Fiyatlar: kategori başına en büyük fonlardan başlayıp n başarılı fona ulaşana kadar
    plan = {k: list(gr.sort_values("Portföy büyüklüğü (TL)", ascending=False).index[:N_ADAY + 2])
            for k, gr in ft.groupby("Kategori")}
    toplam = max(len(plan) * N_ADAY, 1)
    prog = st.progress(0.0, text="TEFAS fiyatları çekiliyor (dakikada ~6 istek sınırı nedeniyle ilk seferde 2-3 dk sürebilir)...")
    bitti = 0
    iptal = False
    for kat, kodlar in plan.items():
        ok_say = 0
        for code in kodlar:
            if ok_say >= N_ADAY or iptal:
                break
            try:
                df = _fiyat_tek(code, s0, e0)
                s = g["to_wide"](df)["price"].get(code)
                if s is None or s.dropna().empty:
                    raise ValueError("boş fiyat serisi")
                cols[code] = s
                titles[code] = str(ft.loc[code, "Fon adı"])
                ok_say += 1
                bitti += 1
                prog.progress(min(bitti / toplam, 1.0))
            except Exception as e:
                fail.append(code)
                hata = hata or str(e)
                if not cols and len(fail) >= 3:           # hiç veri gelmiyorsa dakikalarca beklemeyelim
                    iptal = True
        if iptal:
            break
    prog.empty()
    cand = pd.DataFrame(cols)
    if fail:
        uyar.append("Verisi alınamayan TEFAS fonları: " + ", ".join(fail[:12]) + (" ..." if len(fail) > 12 else "")
                    + (f" · ilk hata: {hata}" if hata else ""))
    if cand.empty:
        uyar.append(f"Kurulu paket sürümleri: tefas-crawler {_surum('tefas-crawler')} "
                    "(fiyat çekimi için 0.6.0 veya üstü gerekir), pytefas " + _surum("pytefas") + ".")
    sinif = {}
    for c in cand.columns:
        s = KAT_SINIF.get(g["fund_category"](titles.get(c, "")))
        if s:
            sinif[c] = s

    # 3) ETF'ler (Yahoo Finance)
    etf_c = {t: ETF_SINIF[k] for t, (_, k) in ETF_EVREN.items() if k in ETF_SINIF}
    etf_c.update({"GLD": "Altın", "IAU": "Altın"})
    etf_ok = True
    try:
        ep = _etf_cache(tuple(etf_c), start, end)
        cand = pd.concat([cand, ep], axis=1).sort_index()
        sinif.update({t: etf_c[t] for t in ep.columns})
    except Exception as e:
        etf_ok = False
        uyar.append(f"ETF adayları alınamadı, yalnızca TEFAS fonları kullanıldı (Yahoo Finance: {e}). "
                    "Streamlit Cloud'un ortak IP'leri Yahoo tarafından sık sık sınırlanır; birkaç dakika sonra deneyin.")
    cand = cand.loc[:, [c for c in cand.columns if c in sinif]]
    if cand.shape[1]:
        nobs = cand.notna().sum()
        cand = cand.loc[:, nobs[nobs >= 60].index]
    hv = {"key": key, "cand": cand, "titles": titles, "sinif": sinif, "uyar": uyar, "etf": set(ETF_EVREN)}
    if cand.shape[1] and not fail and etf_ok:
        st.session_state["akis_havuz"] = hv
    return hv


def _sepet_kur(g, hv, pr, tutar, getiri_w):
    """Profile uygun sepeti seçer. Dönüş: sözlük (bos, sec, eksik, elenen, n) veya None."""
    from profil_panel import FON_MAX, PROFILLER, sec_fonlar
    cand, sinif = hv["cand"], hv["sinif"]
    if cand.shape[1] == 0:
        return None
    et = g["efficiency_table"](cand, g["rf"])
    dt = g["drawdown_table"](cand)
    m = pd.DataFrame({
        "Sinif": pd.Series(sinif), "Sharpe": et["Sharpe oranı"],
        "Vol": et["Yıllık volatilite (σ)"], "Getiri": et["Yıllıklandırılmış getiri (CAGR)"],
        "MDD": pd.to_numeric(dt["Maks. düşüş (MDD)"], errors="coerce")})
    m = m.loc[m.index.isin(cand.columns)].dropna(subset=["Sinif", "Sharpe", "Vol", "MDD"])
    m["Risk"] = [g["risk_degeri"](cand[c])[0] for c in m.index]
    ok = m["Risk"] <= FON_MAX[pr["idx"]]
    elenen = list(m.index[~ok])
    m = m[ok]
    if m.empty:
        return {"bos": True, "elenen": elenen}
    sec, eksik = sec_fonlar(m, PROFILLER[pr["idx"]], N_PICK, getiri_w)
    if sec.empty:
        return {"bos": True, "elenen": elenen}
    sec["Tutar"] = sec["Ağırlık"] * tutar
    return {"bos": False, "sec": sec, "eksik": eksik, "elenen": elenen, "n": len(m)}


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
    out.append(f"**Sepetin kuruluşu:** {tl(c['tutar'])}, **{profil}** profilinizin örnek varlık dağılımına göre "
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
    yontem = (f"**Seçim yöntemi:** {k['n']} aday (TEFAS'ta kategorisinin en büyük fonları ve ETF evreni) arasından, "
              f"TEFAS risk değeri {FON_MAX[pr['idx']]}'i aşanlar elendi; kalanlar kendi sınıfında Sharpe oranı, "
              "düşük volatilite ve düşük maksimum düşüşe göre puanlandı")
    if c["anla"]["getiri_w"] > 0:
        yontem += "; isteğiniz doğrultusunda puana getiri sırası da eklendi, ancak profilinizin risk sınırı aşılmadı"
    out.append(yontem + ".")
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


def yaz_kars(g, c):
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


def _istek_notu(anla, idx, profil):
    """Yazılan istek profille çelişiyorsa kullanıcıyı açıkça bilgilendirir; sepet her zaman profile göre kurulur."""
    o = anla["odak"]
    nt = []
    if "dengeli yapı" in o and idx != 2:
        nt.append(f"'dengeli' ifadesi {profil} profilinizden farklı")
    if "düşük risk" in o and idx >= 3:
        nt.append(f"'düşük risk' isteği {profil} profilinizin risk düzeyinin altında")
    if "yüksek getiri" in o and idx <= 1:
        nt.append(f"'yüksek getiri' isteği {profil} profilinizin risk sınırıyla bağdaşmıyor")
    if not nt:
        return ""
    return ("**Not:** " + "; ".join(nt) + ". Sepet güvenlik için her zaman profilinizin risk sınırlarına göre "
            "kuruldu. Farklı bir risk düzeyi istiyorsanız profil anketini gözden geçirin.")


YAZARLAR = {"sepet": yaz_sepet, "perf": yaz_perf, "risk": yaz_risk, "kars": yaz_kars}


# ---------- Düzenlenebilir ağırlık tablosu ----------
def _editor(g, hv, k, pr, tutar):
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
    ed = st.data_editor(
        df, hide_index=True, use_container_width=True, disabled=["Kod", "Fon", "Sınıf"],
        key=f"bl_ed_{sw['ver']}",
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
def render_bulten(g):
    from gorsel_panel import GORSEL, hero
    st.markdown("### Fon sepeti akışı")
    st.caption("**1)** Ne incelemek istediğinizi yazın · **2)** Başlıkları işaretleyin · "
               "**3)** 'Oluştur'a basın; profilinize uygun sepet grafiklerle ve yazıyla sunulur. "
               "Fon kodu girmeniz gerekmez; fonlar TEFAS ve ETF verisinden seçilir.")
    yazi = st.text_input("Ne incelemek istiyorsun?", key="bl_yazi",
                         placeholder='Örn. "Bana yüksek getirili dengeli bir sepet oluştur"',
                         help="İsteğe bağlı olarak tutar (200 bin TL) yazabilirsiniz. "
                              "Yazmazsanız profil anketindeki tutar kullanılır. Dönem her zaman son 1 yıldır.")
    st.markdown("**Akışta hangi başlıklar olsun?**")
    cols = st.columns(4)
    secili = [k for i, (k, ad) in enumerate(MODULLER) if cols[i].checkbox(ad, key=f"bl_c_{k}", value=True)]

    if not (yazi or "").strip():
        st.info("Ne incelemek istediğinizi yazın.")
        return
    st.button("Oluştur", type="primary", key="bl_btn",
              on_click=lambda: st.session_state.__setitem__("bl_son", (st.session_state.get("bl_yazi") or "").strip()))
    if st.session_state.get("bl_son") != yazi.strip():
        st.info("Yazdıktan sonra 'Oluştur'a basın.")
        return
    if not secili:
        st.info("Akışı görmek için yukarıdan en az bir başlığı işaretleyin.")
        return
    pr = st.session_state.get("profil")
    if not pr:
        st.warning("Sepet profilinize göre kurulur. Önce yukarıdaki 'Yatırımcı profili' anketini doldurup "
                   "'Profilimi hesapla'ya basın.")
        return
    if not g["live"]:
        st.warning("Sepet için canlı TEFAS verisi gerekir.")
        return

    anla = _anla(g, yazi)
    tutar = float(anla["tutar"] or pr["tutar"])
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=1)          # dönem her zaman son 1 yıl
    from profil_panel import FON_MAX, PROFILLER

    with st.spinner("Sepet hazırlanıyor (ilk seferde veri çekimi birkaç dakika sürebilir)..."):
        try:
            hv = _havuz(g, start, end)
        except Exception as e:
            st.error(f"Aday fon verisi alınamadı: {e}")
            return
        k = _sepet_kur(g, hv, pr, tutar, anla["getiri_w"])
    if k is None:
        st.warning("Aday fon verisi alınamadı. Nedenler:")
        for u in hv["uyar"]:
            st.caption(u)
        return
    if k["bos"]:
        st.warning(f"TEFAS risk değeri {FON_MAX[pr['idx']]} veya altında uygun aday bulunamadı.")
        return
    for u in hv["uyar"]:
        st.caption(u)

    hero_slot = st.container()                    # özet kartı en üstte görünsün, hesap tablodan sonra yapılır
    st.divider()
    sec, duz, uyarilar = _editor(g, hv, k, pr, tutar)
    for u in uyarilar:
        st.warning(u)

    amounts = {x: float(a) for x, a in zip(sec["Kod"], sec["Tutar"])}
    rates = {x: (0.0 if x in hv["etf"] else g["auto_stopaj"](hv["titles"].get(x, ""))[0]) for x in amounts}
    res = g["basket_analysis"](hv["cand"], amounts, None, rates, 0.0, 0.0, g["alpha"])
    if res is None:
        st.warning("Seçilen varlıkların ortak tarihlerinde yeterli veri yok. Ağırlıkları değiştirmeyi deneyin.")
        return
    gun = max(int(res["days"]), 1)
    c = {"pr": pr, "hv": hv, "k": k, "sec": sec, "res": res, "tutar": tutar, "anla": anla, "duzenlendi": duz,
         "bt": g["basket_text"](res, f"seçilen aralık ({gun} gün)")}
    with hero_slot:
        try:
            hero(g, c)
        except Exception as e:
            st.caption(f"Özet kartı oluşturulamadı: {e}")

    profil = PROFILLER[pr["idx"]]
    odak = (", ".join(anla["odak"]) + " odaklı, ") if anla["odak"] else ""
    uyum = _istek_notu(anla, pr["idx"], profil)
    st.markdown(f"**İsteğiniz:** “{yazi.strip()}”  \n"
                f"**Akış:** {odak}**{profil}** profilinize uygun, {g['tl'](tutar)} tutarında sepet. "
                f"Veri dönemi {res['start']:%d.%m.%Y} – {res['total'].index[-1]:%d.%m.%Y}.")
    if uyum:
        st.info(uyum)
    for key, ad in MODULLER:
        if key not in secili:
            continue
        st.divider()
        st.markdown(f"### {ad}")
        try:
            GORSEL[key](g, c)
        except Exception as e:
            st.caption(f"Bu bölümün görselleri oluşturulamadı: {e}")
        try:
            paras = YAZARLAR[key](g, c)
        except Exception as e:
            paras = [f"Bu bölüm oluşturulamadı: {e}"]
        st.markdown("##### Yazılı değerlendirme")
        for p_ in paras:
            st.markdown(p_)
    st.caption(DISC)
