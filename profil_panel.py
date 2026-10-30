"""profil_panel.py - Yatırımcı profili, risk analizi ve profile göre örnek portföy önerisi.

app.py'nin globals() sözlüğünü parametre olarak alır (mevcut fonksiyonları yeniden kullanır).
Kural tabanlı, genel bilgilendirme amaçlıdır; kişiye özel yatırım danışmanlığı değildir.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

DISC = ("Bu bölüm kural tabanlı, genel bilgilendirme amaçlı bir örnektir; kişiye özel yatırım "
        "danışmanlığı veya al-sat tavsiyesi değildir. Yatırım danışmanlığı hizmeti SPK'dan "
        "yetkili kuruluşlarca sunulur. Sıralamalar geçmiş verilere dayanır ve gelecekteki "
        "performansın göstergesi değildir. Karar vermeden önce fon izahnamelerini ve KAP "
        "duyurularını inceleyin.")

PROFILLER = ["Muhafazakâr", "Temkinli", "Dengeli", "Büyüme", "Agresif"]
ACIKLAMA = {
    "Muhafazakâr": "Anaparanın korunması önceliklidir; dalgalanmaya çok düşük tolerans vardır. "
                   "Örnek dağılım ağırlıklı olarak para piyasası ve borçlanma araçlarından oluşur.",
    "Temkinli": "Düşük dalgalanma ile enflasyonun üzerinde getiri hedeflenir. Hisse payı sınırlı "
                "tutulur, çoğunluk düşük riskli araçlardadır.",
    "Dengeli": "Getiri ve risk dengelenir. Orta vadede dalgalanmaya katlanılabilir; hisse, "
               "borçlanma ve altın bir arada bulunur.",
    "Büyüme": "Uzun vadede yüksek getiri için belirgin dalgalanma kabul edilir. Hisse ve yabancı "
              "varlıklar ağırlıktadır.",
    "Agresif": "Maksimum getiri için yüksek dalgalanma ve büyük geçici kayıplar kabul edilir. "
               "Portföy ağırlıklı olarak hisse ve yabancı varlıklardır.",
}
SINIFLAR = ["Para piyasası", "Borçlanma araçları", "Karma / değişken", "Hisse senedi (TR)",
            "Yabancı varlık / ETF", "Altın"]
HEDEF = {  # genel, eğitim amaçlı örnek varlık dağılımları (%)
    "Muhafazakâr": [50, 35, 5, 0, 0, 10],
    "Temkinli":    [30, 35, 10, 10, 5, 10],
    "Dengeli":     [15, 25, 10, 25, 10, 15],
    "Büyüme":      [5, 15, 10, 35, 20, 15],
    "Agresif":     [0, 5, 5, 45, 30, 15],
}
HEDEF = {k: dict(zip(SINIFLAR, v)) for k, v in HEDEF.items()}
KAT_SINIF = {  # app.py fund_category() çıktısı -> sınıf
    "Para piyasası": "Para piyasası", "Borçlanma araçları": "Borçlanma araçları",
    "Karma / çoklu varlık": "Karma / değişken", "Hisse senedi": "Hisse senedi (TR)",
    "Yabancı / eurobond": "Yabancı varlık / ETF", "Altın ve kıymetli maden": "Altın",
}
ETF_SINIF = {"ABD hisse": "Yabancı varlık / ETF", "Sektör": "Yabancı varlık / ETF",
             "Uluslararası": "Yabancı varlık / ETF"}
SKOR_AGIRLIK = {  # (Sharpe, düşük volatilite, düşük maks. düşüş)
    "Muhafazakâr": (0.25, 0.40, 0.35), "Temkinli": (0.35, 0.35, 0.30),
    "Dengeli": (0.45, 0.30, 0.25), "Büyüme": (0.55, 0.25, 0.20), "Agresif": (0.65, 0.20, 0.15),
}
AMACLAR = ["Birikim / servet artırma", "Emeklilik", "Ev veya araba alımı", "Çocuk eğitimi",
           "Düzenli gelir", "Kısa vadeli nakit yönetimi"]

SORULAR = [
    dict(id="yas", grup="kap", soru="Yaş aralığınız",
         opts=[("30 yaş altı", 5), ("30-39", 4), ("40-49", 3), ("50-59", 2), ("60 ve üzeri", 1)]),
    dict(id="vade", grup="kap", soru="Yatırımı ne kadar süre elinizde tutmayı planlıyorsunuz?",
         opts=[("1 yıldan kısa", 1), ("1-3 yıl", 2), ("3-5 yıl", 3), ("5-10 yıl", 4),
               ("10 yıldan uzun", 5)]),
    dict(id="gelir", grup="kap", soru="Gelir durumunuz",
         opts=[("Düzensiz veya belirsiz", 1), ("Düzenli ama ihtiyaçlarımı zor karşılıyor", 2),
               ("Düzenli ve yeterli", 4), ("Düzenli, yeterli ve birikim yapabiliyorum", 5)]),
    dict(id="acil", grup="kap", soru="Acil durum birikiminiz (aylık giderlere göre)",
         opts=[("Yok", 1), ("1-3 aylık", 3), ("6 ay ve üzeri", 5)]),
    dict(id="borc", grup="kap", soru="Aylık borç ödemelerinizin gelirinize oranı",
         opts=[("%50 ve üzeri", 1), ("%20-50 arası", 3), ("%20'nin altı veya borcum yok", 5)]),
    dict(id="deney", grup="tol", soru="Yatırım deneyiminiz",
         opts=[("Hiç yok", 1), ("Mevduat ve basit fonlar", 2), ("Hisse ve fonlar (1-3 yıl)", 3),
               ("Hisse, fon, ETF (3 yıldan fazla)", 5)]),
    dict(id="dusus", grup="tol", soru="Yatırımınız kısa sürede %20 değer kaybederse ne yaparsınız?",
         opts=[("Hepsini satarım", 1), ("Bir kısmını satarım", 2), ("Beklerim", 4),
               ("Daha fazla alırım", 5)]),
    dict(id="tercih", grup="tol", soru="Aşağıdakilerden hangisi size en yakın?",
         opts=[("Kaybetmek istemem, getiri düşük olsa da olur", 1),
               ("Düşük getiri, düşük risk", 2), ("Getiri ve risk dengeli olsun", 3),
               ("Yüksek getiri için dalgalanmaya katlanırım", 4),
               ("Maksimum getiri, büyük dalgalanma kabul", 5)]),
    dict(id="kayip", grup="tol", soru="Bir yılda katlanabileceğiniz en yüksek kayıp",
         opts=[("%0, anaparam korunmalı", 1), ("%5'e kadar", 2), ("%10'a kadar", 3),
               ("%20'ye kadar", 4), ("%30 ve üzeri", 5)]),
]


# ---------- Hesaplama (saf fonksiyonlar) ----------
def hesapla(cevap):
    """cevap: {soru_id: seçilen etiket}. Kapasite ve toleransın düşüğüne göre profil."""
    sk = {q["id"]: dict(q["opts"])[cevap[q["id"]]] for q in SORULAR}
    cap = float(np.mean([sk[q["id"]] for q in SORULAR if q["grup"] == "kap"]))
    tol = float(np.mean([sk[q["id"]] for q in SORULAR if q["grup"] == "tol"]))
    idx = int(sum(min(cap, tol) >= t for t in (1.8, 2.6, 3.4, 4.2)))
    limitler, nedenler = [idx], []
    if sk["vade"] == 1:
        limitler.append(1)
        nedenler.append("Yatırım vadeniz 1 yıldan kısa olduğu için profil en fazla Temkinli ile sınırlandı.")
    elif sk["vade"] == 2:
        limitler.append(2)
        nedenler.append("Yatırım vadeniz 1-3 yıl olduğu için profil en fazla Dengeli ile sınırlandı.")
    if sk["acil"] == 1:
        limitler.append(2)
        nedenler.append("Acil durum birikiminiz olmadığı için profil en fazla Dengeli ile sınırlandı.")
    if sk["borc"] == 1:
        limitler.append(2)
        nedenler.append("Borç yükünüz yüksek olduğu için profil en fazla Dengeli ile sınırlandı.")
    return {"skor": sk, "cap": cap, "tol": tol, "idx": int(min(limitler)), "nedenler": nedenler}


def sec_fonlar(m, profil, n_pick):
    """m: index=kod; sütunlar Sinif, Sharpe, Vol, MDD. Sınıf içinde puanlayıp en iyileri seçer.
    Aday olmayan sınıfların payı, aday olan sınıflara orantılı dağıtılır."""
    ws, wv, wd = SKOR_AGIRLIK[profil]
    hedef = {s: w for s, w in HEDEF[profil].items() if w > 0}
    kapsam = {s: w for s, w in hedef.items() if (m["Sinif"] == s).any()}
    eksik = [s for s in hedef if s not in kapsam]
    toplam = sum(kapsam.values())
    rows = []
    for s, w in kapsam.items():
        grp = m[m["Sinif"] == s]
        sc = (ws * grp["Sharpe"].rank(pct=True) + wv * (-grp["Vol"]).rank(pct=True)
              + wd * grp["MDD"].rank(pct=True))
        top = sc.sort_values(ascending=False).head(n_pick).index
        r_sh = grp["Sharpe"].rank(ascending=False)
        r_vo = grp["Vol"].rank(ascending=True)
        r_dd = grp["MDD"].rank(ascending=False)
        for c in top:
            rows.append({
                "Kod": c, "Sınıf": s, "Sınıf hedefi": w / toplam,
                "Ağırlık": w / toplam / len(top),
                "Gerekçe": (f"{len(grp)} aday içinde Sharpe sırası {int(r_sh[c])}, "
                            f"volatilite sırası {int(r_vo[c])}, maks. düşüş sırası {int(r_dd[c])}")})
    return pd.DataFrame(rows), eksik


# ---------- Arayüz: profil ----------
def render_profil(g):
    th = g["th"]
    st.subheader("Yatırımcı profili")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Profilinizi oluşturun**  \nKısa bir anket: risk kapasiteniz ve toleransınız.")
    c2.markdown("**2. Riskinizi görün**  \nProfiliniz ve örnek varlık dağılımı.")
    c3.markdown("**3. Örnek portföy alın**  \nProfilinize uygun fonlar ve örnek sepet.")
    st.caption(DISC)

    with st.form("profil_form"):
        a1, a2 = st.columns(2)
        ad = a1.text_input("Adınız veya takma adınız (isteğe bağlı)")
        amac = a2.selectbox("Yatırım amacınız", AMACLAR)
        tutar = st.number_input("Yatırıma ayırabileceğiniz toplam tutar (TL)", 10000.0, 1e12,
                                100000.0, 10000.0)
        cev = {}
        for q in SORULAR:
            cev[q["id"]] = st.radio(q["soru"], [o[0] for o in q["opts"]], index=None,
                                    key=f"pf_{q['id']}")
        ok = st.form_submit_button("Profilimi hesapla")
    if ok:
        if any(v is None for v in cev.values()):
            st.warning("Lütfen tüm soruları yanıtlayın.")
        else:
            r = hesapla(cev)
            r.update({"ad": ad, "amac": amac, "tutar": float(tutar), "cevap": cev,
                      "tarih": str(pd.Timestamp.today().date())})
            st.session_state["profil"] = r

    pr = st.session_state.get("profil")
    if not pr:
        return
    profil = PROFILLER[pr["idx"]]
    st.divider()
    st.markdown(f"#### {pr['ad'] + ' için ' if pr.get('ad') else ''}profil sonucu")
    k1, k2, k3 = st.columns(3)
    k1.metric("Risk profiliniz", profil)
    k2.metric("Risk kapasitesi (1-5)", f"{pr['cap']:.1f}",
              help="Yaş, vade, gelir, acil birikim ve borç yükünden hesaplanır.")
    k3.metric("Risk toleransı (1-5)", f"{pr['tol']:.1f}",
              help="Deneyim ve düşüşe verdiğiniz tepkiden hesaplanır.")
    st.markdown(ACIKLAMA[profil])
    for n in pr["nedenler"]:
        st.warning(n)
    st.caption("Profil, kapasite ve tolerans skorlarından düşük olana göre belirlenir "
               "(temkinli yaklaşım); vade, acil birikim ve borç yükü üst sınır koyabilir.")
    d1, d2 = st.columns([1, 1])
    hedef = pd.Series(HEDEF[profil])
    hedef = hedef[hedef > 0]
    with d1:
        st.markdown("##### Örnek varlık dağılımı")
        dn = pd.DataFrame({"Sınıf": hedef.index, "Deger": hedef.values.astype(float)})
        st.altair_chart(g["donut_chart"](dn, "Sınıf", "Deger", th), use_container_width=True,
                        theme=None)
    with d2:
        st.markdown("##### Hedef paylar")
        st.dataframe(hedef.rename("Pay (%)").to_frame().style.format("{:.0f}"),
                     use_container_width=True)
        st.caption("Genel, eğitim amaçlı örnek dağılımdır; kişisel durumunuza göre değişebilir.")
    st.download_button("Profil özetini indir (JSON)",
                       json.dumps({**pr, "profil": profil}, ensure_ascii=False, indent=2).encode("utf-8"),
                       "yatirimci_profili.json", "application/json")
    st.info("Profilinize uygun fon önerileri için 'Öneri ve örnek portföy' sekmesine geçin.")


# ---------- Arayüz: öneri ----------
def _aktar(tefas, etfs, amts):
    cur = [c.strip().upper() for c in st.session_state.get("codes_in", "").split(",") if c.strip()]
    for c in tefas:
        if c not in cur:
            cur.append(c)
    st.session_state["codes_in"] = ",".join(cur)
    st.session_state["bk_pick"] = list(tefas)
    st.session_state["bk_etf"] = list(etfs)
    for c in tefas:
        st.session_state[f"bk_amt_{c}"] = float(amts[c])
    for c in etfs:
        st.session_state[f"bk_eamt_{c}"] = float(amts[c])
    st.session_state["aktarildi"] = True


def _pazar_adaylari(g, start, end):
    cur, ref, _, _ = g["load_fund_info"](str(pd.Timestamp.today().date()))
    ft = g["fund_table"](cur, ref)
    ft = ft[ft["Kategori"].isin(KAT_SINIF)]
    top = (ft.sort_values("Portföy büyüklüğü (TL)", ascending=False)
             .groupby("Kategori").head(3))
    cols, ttl, fail = {}, {}, []
    prog = st.progress(0.0)
    for i, code in enumerate(top.index):
        try:
            raw = g["load_tefas"]((code,), str(start), str(end))
            s = g["to_wide"](raw)["price"].get(code)
            if s is None or s.dropna().empty:
                raise ValueError("boş")
            cols[code] = s
            ttl[code] = str(top.loc[code, "Fon adı"])
        except Exception:
            fail.append(code)
        prog.progress((i + 1) / len(top))
    prog.empty()
    return {"prices": pd.DataFrame(cols), "titles": ttl, "fail": fail}


# ---------- Monte Carlo: gelecek senaryoları ----------
VADE_YIL = {"1 yıldan kısa": 1, "1-3 yıl": 3, "3-5 yıl": 5, "5-10 yıl": 10, "10 yıldan uzun": 15}


def mc_simule(mu_log, sigma, yil, aylik, v0, n=5000, seed=42):
    """Aylık adımlarla değer yolları. Aylık log-getiri ~ N(mu/12, sigma^2/12);
    her ay sonunda sabit katkı eklenir. Dönen dizi: (n, adım+1)."""
    steps = int(round(yil * 12))
    rng = np.random.default_rng(seed)
    r = mu_log / 12 + sigma / np.sqrt(12) * rng.standard_normal((n, steps))
    g_ = np.exp(r)
    V = np.empty((n, steps + 1))
    V[:, 0] = v0
    for t in range(steps):
        V[:, t + 1] = V[:, t] * g_[:, t] + aylik
    return V


def _monte_carlo(g, res, pr, tutar):
    import altair as alt
    th, pc, tl = g["th"], g["pc"], g["tl"]
    st.markdown("##### Gelecek senaryoları (Monte Carlo)")
    st.caption("Sepetin geçmiş oynaklığından rastgele 5.000 gelecek yolu üretilir. Bu bir tahmin "
               "değil, olasılık aralığıdır; stopaj ve masraflar dahil değildir.")
    tot = res["total"]
    lr = np.log1p(tot.pct_change().dropna())
    gun = max(int(res["days"]), 1)
    if len(lr) < 20 or not np.isfinite(lr.std()) or lr.std() == 0:
        st.info("Simülasyon için yeterli geçmiş veri yok.")
        return
    ppy = (len(tot) - 1) / (gun / 365.25)
    hist_mu = float(lr.mean() * ppy)
    hist_sig = float(lr.std() * np.sqrt(ppy))
    hist_ret = float(np.expm1(hist_mu))
    st.caption(f"Geçmişten tahmin (seçilen aralık): yıllık bileşik getiri {pc(hist_ret)}, "
               f"yıllık volatilite {pc(hist_sig)}.")

    c1, c2, c3, c4 = st.columns(4)
    vade0 = VADE_YIL.get(pr.get("cevap", {}).get("vade"), 5)
    yil = c1.slider("Süre (yıl)", 1, 30, vade0, key="mc_yil")
    aylik = c2.number_input("Aylık ek katkı (TL)", 0.0, 1e9, 0.0, 1000.0, key="mc_aylik")
    enf = c3.number_input("Yıllık enflasyon varsayımı (%)", 0.0, 200.0, 0.0, 1.0, key="mc_enf",
                          help="0 bırakırsanız tüm tutarlar nominal TL olur.") / 100
    hedef_bugun = c4.number_input("Hedef (bugünkü TL ile)", 10000.0, 1e12, float(tutar * 2),
                                  10000.0, key="mc_hedef")
    yontem = st.radio("Beklenen getiri", ["Geçmiş dönemden", "Elle gir"], horizontal=True,
                      key="mc_yontem")
    if yontem == "Elle gir":
        ret = st.number_input("Beklenen yıllık bileşik getiri (%)", -50.0, 300.0,
                              float(np.clip(round(hist_ret * 100, 1), -50.0, 300.0)), 1.0,
                              key="mc_ret") / 100
        mu_log = float(np.log1p(ret))
    else:
        mu_log = hist_mu
        if hist_ret > 0.30:
            st.warning(f"Geçmiş dönem getirisi ({pc(hist_ret)}) yüksek. Bu, TL'deki yüksek enflasyon "
                       "veya güçlü bir piyasa döneminin sonucu olabilir ve gelecekte tekrarlanmayabilir; "
                       "'Elle gir' ile daha temkinli bir varsayım denemeyi düşünün.")

    v0 = float(res["table"]["Tutar (TL)"].sum())
    V = mc_simule(mu_log, hist_sig, yil, aylik, v0)
    steps = V.shape[1] - 1
    enf_f = (1 + enf) ** yil
    hedef_nom = hedef_bugun * enf_f
    yatirilan = v0 + aylik * steps
    sonlar = V[:, -1]
    p_hedef = float((sonlar >= hedef_nom).mean())
    p_zarar = float((sonlar < yatirilan).mean())
    q = np.percentile(sonlar, [5, 25, 50, 75, 95])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hedefe ulaşma olasılığı", f"%{p_hedef * 100:.0f}")
    m2.metric("Kötü senaryo (%5)", tl(q[0]))
    m3.metric("Medyan senaryo", tl(q[2]))
    m4.metric("İyi senaryo (%95)", tl(q[4]))

    pq = np.percentile(V, [5, 25, 50, 75, 95], axis=0)
    df = pd.DataFrame({"Yıl": np.arange(steps + 1) / 12, "p5": pq[0], "p25": pq[1], "p50": pq[2],
                       "p75": pq[3], "p95": pq[4], "yatirilan": v0 + aylik * np.arange(steps + 1)})
    base = alt.Chart(df).encode(x=alt.X("Yıl:Q", title="Yıl"))
    b90 = base.mark_area(opacity=0.18, color=th["accent"]).encode(
        y=alt.Y("p5:Q", title=None, axis=alt.Axis(format=",.2s")), y2="p95:Q")
    b50 = base.mark_area(opacity=0.35, color=th["accent"]).encode(y="p25:Q", y2="p75:Q")
    med = base.mark_line(color=th["accent"], strokeWidth=3).encode(
        y="p50:Q", tooltip=[alt.Tooltip("Yıl:Q", format=".1f"), alt.Tooltip("p50:Q", format=",.0f")])
    inv = base.mark_line(color=th["muted"], strokeDash=[6, 4], strokeWidth=2).encode(y="yatirilan:Q")
    rule = alt.Chart(pd.DataFrame({"h": [hedef_nom]})).mark_rule(
        color="#d97706", strokeDash=[4, 4], strokeWidth=2).encode(y="h:Q")
    st.altair_chart(g["_style_alt"](b90 + b50 + med + inv + rule, "", th, 340),
                    use_container_width=True, theme=None)
    st.caption("Koyu bant: %25-%75 arası senaryolar, açık bant: %5-%95 arası. Çizgi: medyan. "
               "Kesikli gri: yatırılan toplam tutar. Turuncu kesikli: hedef.")

    with st.expander("Yazılı değerlendirme ve yüzdelik tablosu"):
        st.markdown(
            f"Bu varsayımlarla {yil} yıl sonunda hedefe ({tl(hedef_bugun)} bugünkü TL ile"
            + (f", nominal yaklaşık {tl(hedef_nom)}" if enf > 0 else "")
            + f") ulaşma olasılığı yaklaşık **%{p_hedef * 100:.0f}**. En kötü %5'lik senaryoda sepet "
              f"{tl(q[0])}, medyan senaryoda {tl(q[2])}, en iyi %5'lik senaryoda {tl(q[4])} olur. "
              f"Toplam yatırılan tutarın ({tl(yatirilan)}) altında kalma olasılığı yaklaşık "
              f"**%{p_zarar * 100:.0f}**.")
        tab = pd.DataFrame({"Yüzdelik": ["%5", "%25", "%50 (medyan)", "%75", "%95"],
                            "Son değer (nominal TL)": q, "Bugünkü TL ile": q / enf_f})
        st.dataframe(tab.set_index("Yüzdelik").style.format("{:,.0f}"), use_container_width=True)
    st.caption("Olasılıklar model varsayımlarına bağlıdır: normal dağılım, sabit oynaklık ve "
               "geçmişten tahmin edilen getiri kullanılır. Gerçekte aşırı kayıp riski daha büyük "
               "olabilir; sonuçlar tavsiye veya garanti değildir.")


def render_oneri(g):
    from etf_panel import ETF_EVREN, etf_tl_fiyat
    th, p, rf, alpha = g["th"], g["p"], g["rf"], g["alpha"]
    start, end = g["start"], g["end"]
    st.subheader("Öneri ve örnek portföy")
    st.caption(DISC)
    pr = st.session_state.get("profil")
    if not pr:
        st.info("Önce 'Yatırımcı profili' sekmesinde profilinizi oluşturun.")
        return
    profil = PROFILLER[pr["idx"]]
    st.markdown(f"**Profiliniz: {profil}.** Aday fonlar, profilinize göre ağırlıklandırılmış bir "
                "puanla (Sharpe, volatilite, maksimum düşüş) sınıf içinde sıralanır.")
    o1, o2, o3 = st.columns(3)
    use_etf = o1.checkbox("ETF'leri de aday yap (TL'ye çevrilir)", value=True, key="on_etf",
                          help="ETF'ler USD/TRY kuruyla TL'ye çevrilir; kur riski içerir.")
    n_pick = o2.selectbox("Sınıf başına en fazla fon", [1, 2, 3], index=1, key="on_n")
    tutar = o3.number_input("Toplam tutar (TL)", 10000.0, 1e12, float(pr["tutar"]), 10000.0,
                            key="on_tutar")

    if g["live"]:
        if st.button("Pazardan en büyük fonları aday olarak getir (kategori başına 3)",
                     key="on_pazar"):
            try:
                with st.spinner("Aday fonlar çekiliyor, bu işlem biraz sürebilir..."):
                    st.session_state["on_extra"] = _pazar_adaylari(g, start, end)
            except Exception as e:
                st.error(f"Pazar adayları alınamadı: {e}")
    else:
        st.caption("Pazardan aday getirmek için canlı TEFAS verisi gerekir.")
    extra = st.session_state.get("on_extra")

    # Aday havuzu
    cand, titles = p.copy(), dict(g["titles_all"])
    sinif = {}
    if extra and not extra["prices"].empty:
        yeni = extra["prices"]
        cand = pd.concat([cand, yeni[[c for c in yeni.columns if c not in cand.columns]]], axis=1)
        titles.update(extra["titles"])
        if extra["fail"]:
            st.caption("Verisi alınamayan adaylar atlandı: " + ", ".join(extra["fail"]))
    for c in cand.columns:
        s = KAT_SINIF.get(g["fund_category"](titles.get(c, "")))
        if s:
            sinif[c] = s
    if use_etf:
        etf_c = {t: ETF_SINIF[k] for t, (_, k) in ETF_EVREN.items() if k in ETF_SINIF}
        etf_c.update({"GLD": "Altın", "IAU": "Altın"})
        try:
            ep = etf_tl_fiyat(tuple(etf_c), start, end)
            cand = pd.concat([cand, ep], axis=1).sort_index()
            sinif.update({t: etf_c[t] for t in ep.columns})
        except Exception as e:
            st.warning(f"ETF adayları alınamadı, yalnızca TEFAS fonları kullanılıyor: {e}")
    cand = cand.loc[:, [c for c in cand.columns if c in sinif]]
    nobs = cand.notna().sum()
    cand = cand.loc[:, nobs[nobs >= 60].index]
    if cand.shape[1] == 0:
        st.warning("Sınıflandırılabilen ve yeterli verisi olan aday fon yok. Yan paneldeki fon "
                   "kodlarını artırın veya 'Pazardan aday getir' düğmesini kullanın.")
        return
    et = g["efficiency_table"](cand, rf)
    dt = g["drawdown_table"](cand)
    m = pd.DataFrame({
        "Sinif": pd.Series(sinif), "Sharpe": et["Sharpe oranı"],
        "Vol": et["Yıllık volatilite (σ)"], "Getiri": et["Yıllıklandırılmış getiri (CAGR)"],
        "MDD": pd.to_numeric(dt["Maks. düşüş (MDD)"], errors="coerce")})
    m = m.loc[m.index.isin(cand.columns)].dropna(subset=["Sinif", "Sharpe", "Vol", "MDD"])
    st.caption(f"Aday havuzu: {len(m)} fon/ETF · Veri aralığı: {cand.index[0]:%d.%m.%Y} – "
               f"{cand.index[-1]:%d.%m.%Y}")
    sec, eksik = sec_fonlar(m, profil, n_pick)
    if sec.empty:
        st.warning("Profile uygun aday bulunamadı.")
        return
    if eksik:
        st.warning("Şu sınıflarda aday bulunamadı, payları diğer sınıflara orantılı dağıtıldı: "
                   + ", ".join(eksik) + ". Aday eklemek için fon kodlarını artırın veya pazardan aday getirin.")

    def ad_of(c):
        return titles.get(c) or ETF_EVREN.get(c, (c,))[0]
    sec["Ad"] = sec["Kod"].map(ad_of)
    sec["Tür"] = sec["Kod"].map(lambda c: "ETF" if c in ETF_EVREN else "TEFAS")
    sec["Tutar (TL)"] = sec["Ağırlık"] * tutar
    for k, col in (("Sharpe", "Sharpe"), ("Vol", "Volatilite"), ("MDD", "Maks. düşüş"),
                   ("Getiri", "Yıllık getiri")):
        sec[col] = sec["Kod"].map(m[k])
    tb = sec[["Kod", "Ad", "Tür", "Sınıf", "Ağırlık", "Tutar (TL)", "Sharpe", "Volatilite",
              "Maks. düşüş", "Yıllık getiri", "Gerekçe"]].set_index("Kod")

    st.markdown("##### Önerilen örnek portföy")
    st.dataframe(tb.style.format({"Ağırlık": "{:.1%}", "Tutar (TL)": "{:,.0f}", "Sharpe": "{:.2f}",
                                  "Volatilite": "{:.2%}", "Maks. düşüş": "{:.2%}",
                                  "Yıllık getiri": "{:+.2%}"}, na_rep="-"),
                 use_container_width=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Sınıf dağılımı")
        cl = sec.groupby("Sınıf")["Ağırlık"].sum().rename("Deger").rename_axis("Sınıf").reset_index()
        st.altair_chart(g["donut_chart"](cl, "Sınıf", "Deger", th), use_container_width=True,
                        theme=None)
    with g2:
        st.markdown("##### Fon dağılımı")
        fd = sec[["Kod", "Ağırlık"]].rename(columns={"Ağırlık": "Deger"})
        st.altair_chart(g["donut_chart"](fd, "Kod", "Deger", th), use_container_width=True,
                        theme=None)

    # Geçmiş simülasyon (al-tut): yan panelde seçilen tarih aralığı
    amounts = {c: float(tb.loc[c, "Tutar (TL)"]) for c in tb.index}
    rates = {c: (0.0 if c in ETF_EVREN else g["auto_stopaj"](titles.get(c, ""))[0]) for c in amounts}
    res = g["basket_analysis"](cand, amounts, None, rates, 0.0, 0.0, alpha)
    st.markdown("##### Seçilen aralıkta bu sepet ne kazandırır veya kaybettirirdi? (geçmiş simülasyon)")
    if res is None:
        st.info("Seçilen fonların ortak tarihlerinde simülasyon için yeterli veri yok.")
    else:
        t = res["table"]
        toplam = t["Tutar (TL)"].sum()
        net = t["Net değer (TL)"].sum()
        bitis = res["total"].index[-1]
        gun = max(int(res["days"]), 1)
        st.caption(f"Simülasyon dönemi: {res['start']:%d.%m.%Y} – {bitis:%d.%m.%Y} ({gun} gün). "
                   "Fonların ortak veri tarihleri kullanılır; sepette sonradan kurulan bir fon "
                   "varsa dönem yan paneldeki aralıktan kısa olabilir.")
        s1, s2, s3 = st.columns(3)
        s1.metric("Brüt getiri", g["pc"](res["ret"]))
        s2.metric("Net kâr/zarar", g["signed_tl"](net - toplam), f"{(net / toplam - 1) * 100:+.2f}%")
        s3.metric("Net elde edilen", g["tl"](net))
        s4, s5, s6 = st.columns(3)
        yil = (net / toplam) ** (365.25 / gun) - 1 if gun >= 90 and net > 0 else None
        s4.metric("Yıllıklandırılmış net getiri", "-" if yil is None else g["pc"](yil),
                  help="Yalnızca 90 günden uzun dönemlerde ve pozitif sonuçta gösterilir.")
        s5.metric("Yıllık volatilite", g["pc"](res["vol"]))
        s6.metric("Maks. düşüş", g["pc"](res["dd"]))
        with st.expander("Fon bazında kâr/zarar"):
            ft_ = t[["Tutar (TL)", "Dönem getirisi", "Stopaj (TL)", "Net kâr/zarar (TL)"]]
            st.dataframe(ft_.style.format({"Tutar (TL)": "{:,.0f}", "Dönem getirisi": "{:+.2%}",
                                           "Stopaj (TL)": "{:,.0f}",
                                           "Net kâr/zarar (TL)": "{:+,.0f}"}),
                         use_container_width=True)
        with st.expander("Yazılı değerlendirme"):
            for para in g["basket_text"](res, f"seçilen aralık ({gun} gün)"):
                st.markdown(para)
        st.caption("Stopaj, TEFAS fonları için fon adından tahmin edilmiştir; ETF için vergi ve "
                   "komisyon 0 varsayılmıştır. Portföy sepeti sekmesinde değiştirebilirsiniz.")
        _monte_carlo(g, res, pr, tutar)

    tefas = [c for c in tb.index if c not in ETF_EVREN]
    etfs = [c for c in tb.index if c in ETF_EVREN]
    st.button("Önerilen sepeti 'Portföy sepeti' sekmesine aktar", key="on_aktar",
              on_click=_aktar, args=(tefas, etfs, amounts))
    if st.session_state.pop("aktarildi", False):
        st.success("Aktarıldı. 'Portföy sepeti' sekmesine geçip stopaj ve masraf ayarlarını "
                   "düzenleyebilirsiniz. Fon kodları yan paneldeki listeye de eklendi.")
    st.download_button("Öneriyi indir (CSV)", tb.to_csv().encode("utf-8-sig"),
                       "ornek_portfoy.csv", "text/csv")
