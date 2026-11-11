"""gorsel_panel.py - 'Oluştur' sonrası görsel ekran.

bulten_panel.py bu modülü çağırır: önce özet (hero) kartı, sonra işaretlenen her başlık için
grafikler/kartlar, ardından yazılı akış. app.py'nin globals() sözlüğünü (g) ve bulten_panel'in
bağlam sözlüğünü (c) alır. Hiçbir hesap burada yapılmaz; app.py'deki fonksiyonlar kullanılır.
"""
import html as _h

import numpy as np
import pandas as pd
import streamlit as st

RISK_RENK = ["#16a34a", "#4ade80", "#a3e635", "#facc15", "#fb923c", "#ef4444", "#b3402a"]



# ---------- Streamlit sürüm uyumluluğu ----------
def stretch(fn, *a, **k):
    """use_container_width yeni Streamlit'te kaldırıldı (width='stretch' geldi); sırayla dener."""
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return fn(*a, **k, **kw)
        except TypeError:
            continue
    return fn(*a, **k)


def _ac(ch):
    for kw in ({"theme": None}, {}):
        try:
            return stretch(st.altair_chart, ch, **kw)
        except TypeError:
            continue


def _ciz(builder):
    """Grafiği kurar ve çizer; hata olursa sessizce geçmek yerine görünür uyarı verir."""
    try:
        _ac(builder())
    except Exception as e:
        st.warning(f"Grafik oluşturulamadı: {type(e).__name__}: {str(e)[:160]}")


CSS = """
<style>
.hero{background:linear-gradient(135deg,#c96442 0%,#7d3420 100%);color:#fff;border-radius:16px;padding:20px 22px;margin:6px 0 16px}
.hero h2{color:#fff !important;margin:0 0 2px;font-size:22px}
.hero .sub{opacity:.88;font-size:13px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.kpi{background:rgba(255,255,255,.16);border-radius:12px;padding:10px 12px}
.kpi .l{font-size:11px;letter-spacing:.6px;text-transform:uppercase;opacity:.85}
.kpi .v{font-size:20px;font-weight:700;margin-top:2px}
.kpi .s{font-size:12px;opacity:.9}
.fc-wrap{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin:8px 0 16px}
.fc{background:#fff;border:1px solid #e8e2d6;border-left:5px solid #c96442;border-radius:12px;padding:12px 14px;box-shadow:0 1px 3px rgba(16,24,40,.08)}
.fc.up{border-left-color:#2f7d4f}.fc.down{border-left-color:#b3402a}
.fc .top{display:flex;justify-content:space-between;align-items:center}
.fc .code{font-size:17px;font-weight:700;color:#111827}
.fc .nm{font-size:11px;color:#6b7280;margin:1px 0 6px;min-height:28px}
.tag{font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px;background:#fbe9e2;color:#c96442;letter-spacing:.4px}
.tag.etf{background:#fef3c7;color:#b45309}
.wb{height:6px;border-radius:99px;background:#e5e7eb;margin:8px 0 4px;overflow:hidden}
.wb i{display:block;height:100%;background:linear-gradient(90deg,#c96442,#d4a24c)}
.wl{font-size:11px;color:#6b7280;display:flex;justify-content:space-between}
.chips2{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}
.ch{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;background:#f3f4f6;color:#4b5563}
.ch.up{background:#dcfce7;color:#2f7d4f}.ch.down{background:#fee2e2;color:#b3402a}
.ok{font-size:11px;font-weight:700;color:#2f7d4f;margin-top:8px}.bad{font-size:11px;font-weight:700;color:#b3402a;margin-top:8px}
.rs{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;margin:10px 0 4px}
.rb{border-radius:10px;padding:10px 0;text-align:center;font-weight:700;color:#fff;opacity:.35;font-size:15px}
.rb.in{opacity:.75}.rb.me{opacity:1;outline:3px solid #111827;transform:scale(1.08)}
.rnote{font-size:12px;color:#6b7280}
</style>
"""


def _chip(label, x):
    if x is None or pd.isna(x):
        return f"<span class='ch'>{label} -</span>"
    cls = "up" if x >= 0 else "down"
    return f"<span class='ch {cls}'>{label} {'▲' if x >= 0 else '▼'}{abs(x) * 100:.1f}%</span>"


def _spark(s, up):
    s = s.dropna()
    if len(s) < 3:
        return ""
    idx = np.unique(np.linspace(0, len(s) - 1, 40).astype(int))
    v = s.iloc[idx].values.astype(float)
    lo, hi = v.min(), v.max()
    rng = (hi - lo) or 1.0
    xs = np.linspace(0, 120, len(v))
    ys = 34 - (v - lo) / rng * 30
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    col = "#2f7d4f" if up else "#b3402a"
    return (f"<svg viewBox='0 0 120 38' preserveAspectRatio='none' style='width:100%;height:38px'>"
            f"<polygon points='0,38 {pts} 120,38' fill='{col}' opacity='.12'/>"
            f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2'/></svg>")


# ---------- Ortak veri ----------
def _veri(g, c):
    if "gv" in c:
        return c["gv"]
    res, hv = c["res"], c["hv"]
    funds = list(res["table"].index)
    P = hv["cand"][funds].dropna()
    P2 = P.copy()
    P2.insert(0, "Sepet", res["total"].reindex(P.index))
    c["gv"] = {"funds": funds, "P": P, "P2": P2}
    return c["gv"]


def _guvenli(f, *a):
    try:
        f(*a)
    except Exception as e:
        st.caption(f"Bu görsel oluşturulamadı: {e}")


# ---------- Özet (hero) ----------
def hero(g, c):
    from profil_panel import PROFILLER, SEPET_ARALIK
    st.markdown(CSS, unsafe_allow_html=True)
    tl, pc = g["tl"], g["pc"]
    res, pr = c["res"], c["pr"]
    t = res["table"]
    tot, net = t["Tutar (TL)"].sum(), t["Net değer (TL)"].sum()
    rv, _ = g["risk_degeri"](res["total"])
    lo, hi = SEPET_ARALIK[pr["idx"]]
    uyum = "—" if pd.isna(rv) else ("Aralıkta" if lo <= rv <= hi else ("Aralık altı" if rv < lo else "Aralık üstü"))
    kpis = [
        ("Net sonuç", signed(net - tot, tl), f"{pc(net / tot - 1)} (1 yıl, stopaj sonrası)"),
        ("Yıllık volatilite", pc(res["vol"]), "fiyat dalgalanması"),
        ("En büyük düşüş", pc(res["dd"]), f"≈ {tl(abs(res['dd']) * tot)}"),
        ("Risk değeri", "-" if pd.isna(rv) else f"{int(rv)} / 7", f"{PROFILLER[pr['idx']]} aralığı {lo}-{hi}: {uyum}"),
    ]
    tiles = "".join(f"<div class='kpi'><div class='l'>{a}</div><div class='v'>{b}</div><div class='s'>{d}</div></div>"
                    for a, b, d in kpis)
    st.markdown(
        f"<div class='hero'><h2>Örnek sepetiniz hazır</h2>"
        f"<div class='sub'>{_h.escape(PROFILLER[pr['idx']])} profil · {tl(c['tutar'])} · "
        f"{res['start']:%d.%m.%Y} – {res['total'].index[-1]:%d.%m.%Y} · {len(t)} varlık · bilgilendirme amaçlı simülasyon, tavsiye değildir</div>"
        f"<div class='kpis'>{tiles}</div></div>", unsafe_allow_html=True)


def signed(x, tl):
    return ("+" if x >= 0 else "-") + tl(abs(x))


# ---------- Portföy sepeti ----------
def gor_sepet(g, c):
    from profil_panel import FON_MAX
    th, tl = g["th"], g["tl"]
    sec, hv, res, pr = c["sec"], c["hv"], c["res"], c["pr"]
    v = _veri(g, c)
    d1, d2 = st.columns(2)
    with d1:
        sn = sec.groupby("Sınıf")["Ağırlık"].sum().rename("Deger").rename_axis("Sınıf").reset_index()
        st.markdown("##### Varlık sınıfı dağılımı")
        _ciz(lambda: g["donut_chart"](sn, "Sınıf", "Deger", th))
    with d2:
        fd = sec[["Kod", "Ağırlık"]].rename(columns={"Ağırlık": "Deger"})
        st.markdown("##### Fon dağılımı")
        _ciz(lambda: g["donut_chart"](fd, "Kod", "Deger", th))

    st.markdown("##### Fonların durumu")
    P = v["P"]
    cards = []
    for _, r in sec.sort_values("Ağırlık", ascending=False).iterrows():
        code = r["Kod"]
        if code not in P.columns:
            continue
        s = P[code]
        tot_r = s.iloc[-1] / s.iloc[0] - 1
        m_ref = s.asof(s.index[-1] - pd.DateOffset(months=1))
        m1 = s.iloc[-1] / m_ref - 1 if pd.notna(m_ref) else np.nan
        rd = g["risk_degeri"](hv["cand"][code].dropna())[0]
        ok = (not pd.isna(rd)) and rd <= FON_MAX[pr["idx"]]
        ad = hv["titles"].get(code)
        if not ad:
            from etf_panel import ETF_EVREN
            ad = ETF_EVREN.get(code, (code,))[0]
        etf = code in hv["etf"]
        cards.append(
            f"<div class='fc {'up' if tot_r >= 0 else 'down'}'>"
            f"<div class='top'><span class='code'>{_h.escape(code)}</span>"
            f"<span class='tag {'etf' if etf else ''}'>{'ETF' if etf else 'TEFAS'}</span></div>"
            f"<div class='nm'>{_h.escape(str(ad)[:60])}</div>"
            f"{_spark(s, tot_r >= 0)}"
            f"<div class='wb'><i style='width:{min(r['Ağırlık'] * 100 * 2.2, 100):.0f}%'></i></div>"
            f"<div class='wl'><span>{_h.escape(r['Sınıf'])}</span><span><b>{r['Ağırlık'] * 100:.1f}%</b> · {tl(r['Tutar'])}</span></div>"
            f"<div class='chips2'>{_chip('1Y', tot_r)}{_chip('1A', m1)}"
            f"<span class='ch'>Risk {'-' if pd.isna(rd) else int(rd)}/7</span></div>"
            f"<div class='{'ok' if ok else 'bad'}'>{'✓ Profil sınırı içinde' if ok else '⚠ Profil sınırının üstünde'}</div></div>")
    st.markdown("<div class='fc-wrap'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


# ---------- Performans ----------
def gor_perf(g, c):
    th, tl = g["th"], g["tl"]
    v = _veri(g, c)
    P2, res = v["P2"], c["res"]
    base = P2 / P2.iloc[0] * 100
    st.markdown("##### Sepet ve fonlar (başlangıç = 100)")
    _ciz(lambda: g["line_chart"](base, th, "", ",.0f", 280))
    ret = (P2.iloc[-1] / P2.iloc[0] - 1).rename("Getiri").rename_axis("Fon").reset_index()
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("##### 1 yıllık getiri")
        _ciz(lambda: g["bar_pos_neg"](ret, "Fon", "Getiri", "", th))
    with b2:
        st.markdown("##### Sepet değeri, fon bazında (TL)")
        _ciz(lambda: g["stacked_area"](res["value"], th, "", ".2s"))
    st.markdown("##### Aylık getiri ısı haritası")
    mr = g["monthly_returns"](P2)
    if len(mr):
        _ciz(lambda: g["monthly_heatmap"](mr, th))


# ---------- Risk ----------
def _risk_olcegi(g, c):
    from profil_panel import PROFILLER, SEPET_ARALIK
    rv, _ = g["risk_degeri"](c["res"]["total"])
    lo, hi = SEPET_ARALIK[c["pr"]["idx"]]
    kutu = []
    for i in range(1, 8):
        cls = "rb" + (" in" if lo <= i <= hi else "") + (" me" if (not pd.isna(rv) and int(rv) == i) else "")
        kutu.append(f"<div class='{cls}' style='background:{RISK_RENK[i - 1]}'>{i}</div>")
    st.markdown("##### Sepetin risk ölçeği (TEFAS 1-7)")
    st.markdown("<div class='rs'>" + "".join(kutu) + "</div>"
                f"<div class='rnote'>Siyah çerçeve: sepetiniz · Koyu renkli kutular: {PROFILLER[c['pr']['idx']]} profilin referans aralığı ({lo}-{hi})</div>",
                unsafe_allow_html=True)


def _risk_katki(g, c):
    import altair as alt
    th = g["th"]
    t = c["res"]["table"]
    long = pd.concat([
        pd.DataFrame({"Fon": t.index, "Tür": "Tutar payı", "Pay": t["Ağırlık"].values}),
        pd.DataFrame({"Fon": t.index, "Tür": "Risk payı", "Pay": t["Risk katkısı"].values})]).dropna()
    ch = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Fon:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="Tür:N",
        y=alt.Y("Pay:Q", title=None, axis=alt.Axis(format=".0%")),
        color=alt.Color("Tür:N", scale=alt.Scale(domain=["Tutar payı", "Risk payı"],
                                                 range=[th["palette"][0], th["down"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Fon", "Tür", alt.Tooltip("Pay:Q", format=".1%")])
    _ciz(lambda: g["_style_alt"](ch, "", th, 230))


def gor_risk(g, c):
    th, rf, alpha = g["th"], g["rf"], g["alpha"]
    v = _veri(g, c)
    P, P2 = v["P"], v["P2"]
    _risk_olcegi(g, c)
    rt = g["risk_table"](P2, "Sepet", rf, alpha).astype(float)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("##### Risk / getiri haritası")
        if not rt.empty:
            sc = (rt[["Yıllık volatilite", "Yıllık getiri"]].rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
                  .rename_axis("Fon").reset_index())
            _ciz(lambda: g["scatter_chart"](sc, th, ""))
    with r2:
        st.markdown("##### Tutar payı ve risk payı")
        _guvenli(_risk_katki, g, c)
    st.markdown("##### Zirveden düşüş (drawdown)")
    Pf = P2.ffill()
    _ciz(lambda: g["line_chart"](Pf / Pf.cummax() - 1, th, "", ".0%", 240))
    if P.shape[1] >= 2:
        st.markdown("##### Fonlar birbirine ne kadar benziyor? (korelasyon)")
        _ciz(lambda: g["corr_heatmap"](g["returns_df"](P).corr(), th))


# ---------- Yatırımcı karşılaştırma ----------
def gor_kars(g, c):
    th, rf = g["th"], g["rf"]
    v = _veri(g, c)
    res, hv = c["res"], c["hv"]
    funds = v["funds"]
    t = res["table"]
    tot, net = t["Tutar (TL)"].sum(), t["Net değer (TL)"].sum()
    rates = {x: (0.0 if x in hv["etf"] else g["auto_stopaj"](hv["titles"].get(x, ""))[0]) for x in funds}
    m = g["investor_metrics"](v["P"], funds, c["tutar"], None, rf, rates, 0.0, 0.0)
    if m.empty:
        st.caption("Karşılaştırma için yeterli veri yok.")
        return
    gun = max(int(res["days"]), 1)
    a = m["Net getiri"].astype(float).rename("Getiri")
    a["Sepet"] = net / tot - 1
    a["Risksiz oran"] = (1 + rf) ** (gun / 365.25) - 1
    b = m["Net kâr/zarar (TL)"].astype(float).rename("Sonuc")
    b["Sepet (toplam tutar)"] = (net - tot) * (c["tutar"] / tot)
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"##### Net getiri (stopaj sonrası, her biri {g['tl'](c['tutar'])})")
        _ciz(lambda: g["bar_pos_neg"](a.rename_axis("Fon").reset_index(), "Fon", "Getiri", "", th))
    with k2:
        st.markdown("##### Net kâr / zarar (TL)")
        _ciz(lambda: g["bar_pos_neg"](b.rename_axis("Fon").reset_index(), "Fon", "Sonuc", "", th, ".2s"))


GORSEL = {"sepet": gor_sepet, "perf": gor_perf, "risk": gor_risk, "kars": gor_kars}
