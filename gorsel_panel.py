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

.pstrip{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:10px;margin:6px 0 14px}
.pk{background:#fff;border:1px solid #e8e2d6;border-top:4px solid #c96442;border-radius:12px;padding:10px 12px}
.pk.up{border-top-color:#2f7d4f}.pk.down{border-top-color:#b3402a}
.pk .l{font-size:10.5px;color:#6f6a60;text-transform:uppercase;letter-spacing:.5px}
.pk .v{font-size:20px;font-weight:700;color:#1f1e1d;margin-top:2px}.pk .s{font-size:11.5px;color:#6f6a60}
.cbar{display:flex;height:34px;border-radius:10px;overflow:hidden;margin:6px 0 6px;font-size:12px;font-weight:700;color:#fff}
.cbar div{display:flex;align-items:center;justify-content:center;min-width:0;overflow:hidden}
.cleg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#6f6a60;margin-bottom:10px}
.cleg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
.lbwrap{background:#fff;border:1px solid #e8e2d6;border-radius:14px;padding:8px 14px;margin-bottom:10px}
.lb{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #f1ede4}.lb:last-child{border-bottom:0}
.lb .rk{width:22px;height:22px;border-radius:50%;background:#f1ede4;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center}
.lb .nm{width:64px;font-weight:700}.lb .bar{flex:1;height:10px;background:#f1ede4;border-radius:99px;overflow:hidden}
.lb .bar i{display:block;height:100%;border-radius:99px}.lb .vl{width:64px;text-align:right;font-weight:700;font-size:13px}
.tw{overflow-x:auto;margin:6px 0 10px}
.tbl{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e8e2d6;border-radius:12px;font-size:13px}
.tbl th,.tbl td{padding:9px 12px;text-align:right;border-bottom:1px solid #f1ede4;white-space:nowrap}
.tbl th{color:#6f6a60;font-weight:600;font-size:12px;background:#faf7f1}.tbl td:first-child,.tbl th:first-child{text-align:left}
.cmp tr th:first-child{background:#faf7f1}.tbl td.best{color:#2f7d4f;font-weight:700;background:#eef7f1}
.rz{display:inline-block;min-width:24px;padding:2px 8px;border-radius:8px;color:#fff;font-weight:700;text-align:center}
.bkgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:8px 0 16px}
.bk{background:#fff;border:1px solid #e8e2d6;border-radius:14px;padding:14px}
.bk.me{border:2px solid #c96442;background:#fffaf7}
.bk .bt{font-family:'Source Serif 4',Georgia,serif;font-size:17px;font-weight:700;display:flex;justify-content:space-between;align-items:center;gap:6px}
.bk .bs{font-size:12px;color:#6f6a60;margin:2px 0 6px}.bk .bf{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}
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


# ---------- Ortak parçalar ----------
def _strip(items):
    """Parametre şeridi: (etiket, değer, alt not, ton) kartları."""
    cells = "".join(f"<div class='pk {t}'><div class='l'>{a}</div><div class='v'>{b}</div><div class='s'>{d}</div></div>"
                    for a, b, d, t in items)
    st.markdown(f"<div class='pstrip'>{cells}</div>", unsafe_allow_html=True)


def _tone(x):
    return "" if x is None or pd.isna(x) else ("up" if x >= 0 else "down")


def _renk(i):
    pal = ["#c96442", "#2f6f6d", "#d4a24c", "#5b6abf", "#a34a6b", "#3f8f5f", "#6f6a60", "#b5543a"]
    return pal[i % len(pal)]


def _rozet(rd):
    if rd is None or pd.isna(rd):
        return "<span class='rz'>-</span>"
    return f"<span class='rz' style='background:{RISK_RENK[int(rd) - 1]}'>{int(rd)}</span>"


# ---------- 1) Portföy sepeti ----------
def gor_sepet(g, c):
    from profil_panel import FON_MAX
    th, tl, pc = g["th"], g["tl"], g["pc"]
    sec, hv, res, pr = c["sec"], c["hv"], c["res"], c["pr"]
    v = _veri(g, c)
    sn = sec.groupby("Sınıf")["Ağırlık"].sum().sort_values(ascending=False)
    etf_n = sum(1 for x in sec["Kod"] if x in hv["etf"])
    top = sec.sort_values("Ağırlık", ascending=False).iloc[0]
    _strip([("Varlık sayısı", str(len(sec)), f"{len(sec) - etf_n} TEFAS fonu · {etf_n} ETF", ""),
            ("Varlık sınıfı", str(len(sn)), ", ".join(sn.index[:2]) + (" ..." if len(sn) > 2 else ""), ""),
            ("En büyük pay", pc(top["Ağırlık"]), f"{top['Kod']} · {tl(top['Tutar'])}", ""),
            ("Toplam tutar", tl(c["tutar"]), "bilgilendirme amaçlı simülasyon", "")])
    # ağırlık şeridi
    seg = "".join(f"<div style='width:{w * 100:.1f}%;background:{_renk(i)}' title='{_h.escape(s)}'>{w * 100:.0f}%</div>"
                  for i, (s, w) in enumerate(sn.items()))
    leg = "".join(f"<span><i style='background:{_renk(i)}'></i>{_h.escape(s)}</span>" for i, s in enumerate(sn.index))
    st.markdown(f"<div class='cbar'>{seg}</div><div class='cleg'>{leg}</div>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Varlık sınıfı dağılımı")
        sdf = sn.rename("Deger").rename_axis("Sınıf").reset_index()
        _ciz(lambda: g["donut_chart"](sdf, "Sınıf", "Deger", th))
    with d2:
        st.markdown("##### Fon dağılımı")
        fd = sec[["Kod", "Ağırlık"]].rename(columns={"Ağırlık": "Deger"})
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


# ---------- 2) Performans ----------
def gor_perf(g, c):
    th, tl, pc = g["th"], g["tl"], g["pc"]
    v = _veri(g, c)
    P2, res = v["P2"], c["res"]
    t = res["table"]
    tot, net = t["Tutar (TL)"].sum(), t["Net değer (TL)"].sum()
    days = max(int(res["days"]), 1)
    cagr = (1 + res["ret"]) ** (365.25 / days) - 1
    mr = g["monthly_returns"](P2)
    ms = mr["Sepet"].dropna() if "Sepet" in mr else pd.Series(dtype=float)
    items = [("Brüt getiri", pc(res["ret"]), f"{days} gün", _tone(res["ret"])),
             ("Net getiri", pc(net / tot - 1), "stopaj ve masraf sonrası", _tone(net - tot)),
             ("Yıllıklandırılmış", pc(cagr), "bileşik büyüme (CAGR)", _tone(cagr))]
    if len(ms):
        items += [("En iyi ay", pc(ms.max()), str(ms.idxmax()), "up"),
                  ("En kötü ay", pc(ms.min()), str(ms.idxmin()), "down"),
                  ("Pozitif ay oranı", f"%{(ms > 0).mean() * 100:.0f}", f"{int((ms > 0).sum())}/{len(ms)} ay", "")]
    _strip(items)

    st.markdown("##### Sepet ve fonlar (başlangıç = 100)")
    base = P2 / P2.iloc[0] * 100
    _ciz(lambda: g["line_chart"](base, th, "", ",.0f", 300))

    st.markdown("##### Fon performans sıralaması")
    ret = (P2.drop(columns="Sepet").iloc[-1] / P2.drop(columns="Sepet").iloc[0] - 1).sort_values(ascending=False)
    mx = max(float(ret.abs().max()), 1e-9)
    rows = []
    for i, (code, x) in enumerate(ret.items(), 1):
        col = "#2f7d4f" if x >= 0 else "#b3402a"
        rows.append(f"<div class='lb'><span class='rk'>{i}</span><span class='nm'>{_h.escape(code)}</span>"
                    f"<div class='bar'><i style='width:{abs(x) / mx * 100:.0f}%;background:{col}'></i></div>"
                    f"<span class='vl' style='color:{col}'>{'+' if x >= 0 else ''}{x * 100:.1f}%</span></div>")
    st.markdown("<div class='lbwrap'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("##### Sepetin aylık getirileri")
        if len(ms):
            mdf = ms.rename("Getiri").rename_axis("Fon").reset_index()
            _ciz(lambda: g["bar_pos_neg"](mdf, "Fon", "Getiri", "", th))
    with b2:
        st.markdown("##### Sepet değeri, fon bazında (TL)")
        _ciz(lambda: g["stacked_area"](res["value"], th, "", ".2s"))
    st.markdown("##### Aylık getiri ısı haritası")
    if len(mr):
        _ciz(lambda: g["monthly_heatmap"](mr, th))


# ---------- 3) Risk ----------
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
    return g["_style_alt"](ch, "", th, 240)


def gor_risk(g, c):
    from profil_panel import SEPET_ARALIK
    th, rf, alpha, pc, tl = g["th"], g["rf"], g["alpha"], g["pc"], g["tl"]
    v = _veri(g, c)
    P, P2, res, hv = v["P"], v["P2"], c["res"], c["hv"]
    t = res["table"]
    tot = t["Tutar (TL)"].sum()
    rt = g["risk_table"](P2, "Sepet", rf, alpha).astype(float)
    rv, _ = g["risk_degeri"](res["total"])
    lo, hi = SEPET_ARALIK[c["pr"]["idx"]]
    ur = "-" if pd.isna(rv) else ("Aralıkta" if lo <= rv <= hi else ("Aralık altı" if rv < lo else "Aralık üstü"))
    vk = [k for k in rt.columns if k.startswith("VaR")]
    sh = rt.loc["Sepet", "Sharpe"] if "Sepet" in rt.index else np.nan
    _strip([("Risk değeri", "-" if pd.isna(rv) else f"{int(rv)} / 7", f"profil aralığı {lo}-{hi}: {ur}", ""),
            ("Volatilite", pc(res["vol"]), "yıllık", ""),
            ("En büyük düşüş", pc(res["dd"]), f"≈ {tl(abs(res['dd']) * tot)}", "down"),
            (f"VaR %{int(alpha * 100)} (1 gün)", pc(res["var"]), f"≈ {tl(res['var'] * tot)}", "down"),
            ("CVaR (1 gün)", pc(res["cvar"]), "kötü günlerin ortalaması", "down"),
            ("Sharpe", "-" if pd.isna(sh) else f"{sh:.2f}", "risksiz orana göre", ""),
            ("Çeşitlendirme", "-" if pd.isna(res["div"]) else f"{res['div']:.2f}", "1'den büyük = iyi", "")])
    _risk_olcegi(g, c)

    st.markdown("##### Risk karnesi (fon bazında)")
    rows = []
    for code in t.sort_values("Risk katkısı", ascending=False).index:
        rd = g["risk_degeri"](hv["cand"][code].dropna())[0]
        r = rt.loc[code] if code in rt.index else None
        rows.append("<tr><td><b>{}</b></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _h.escape(code), _rozet(rd), pc(t.loc[code, "Ağırlık"]),
            pc(r["Yıllık volatilite"]) if r is not None else "-",
            pc(r["Maks. düşüş"]) if r is not None else "-",
            "-" if r is None or pd.isna(r["Sharpe"]) else f"{r['Sharpe']:.2f}",
            pc(t.loc[code, "Risk katkısı"]) if pd.notna(t.loc[code, "Risk katkısı"]) else "-"))
    st.markdown("<div class='tw'><table class='tbl'><tr><th>Fon</th><th>Risk</th><th>Tutar payı</th><th>Volatilite</th>"
                "<th>Maks. düşüş</th><th>Sharpe</th><th>Risk payı</th></tr>" + "".join(rows) + "</table></div>",
                unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("##### Risk / getiri haritası")
        if not rt.empty:
            sc = (rt[["Yıllık volatilite", "Yıllık getiri"]].rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
                  .rename_axis("Fon").reset_index())
            _ciz(lambda: g["scatter_chart"](sc, th, ""))
    with r2:
        st.markdown("##### Tutar payı ve risk payı")
        _ciz(lambda: _risk_katki(g, c))
    st.markdown("##### Zirveden düşüş (drawdown)")
    Pf = P2.ffill()
    _ciz(lambda: g["line_chart"](Pf / Pf.cummax() - 1, th, "", ".0%", 250))
    if P.shape[1] >= 2:
        st.markdown("##### Fonlar birbirine ne kadar benziyor? (korelasyon)")
        _ciz(lambda: g["corr_heatmap"](g["returns_df"](P).corr(), th))


# ---------- 4) Yatırımcı karşılaştırma ----------
def _bsk_sat(g, key, vals, fmt, better=None):
    """Karşılaştırma tablosu satırı; better='high'/'low' ise en iyi değer vurgulanır."""
    ok = [x for x in vals if x is not None and not pd.isna(x)]
    best = (max(ok) if better == "high" else min(ok)) if ok and better else None
    tds = []
    for x in vals:
        if x is None or pd.isna(x):
            tds.append("<td>-</td>")
        else:
            cls = " class='best'" if best is not None and x == best else ""
            tds.append(f"<td{cls}>{fmt(x)}</td>")
    return f"<tr><th>{key}</th>{''.join(tds)}</tr>"


def gor_kars(g, c):
    from profil_panel import PROFILLER
    th, rf, alpha, pc, tl = g["th"], g["rf"], g["alpha"], g["pc"], g["tl"]
    ana = {"ad": "Önerilen sepet", "aciklama": f"{PROFILLER[c['pr']['idx']]} profil · sizin fonlarınız",
           "sec": c["sec"], "res": c["res"], "ana": True}
    sepetler = [ana] + list(c.get("alts", []))
    if len(sepetler) < 2:
        st.info("Örnek sepet oluşturmak için havuzda, önerilen sepettekilerin dışında yeterli fon bulunamadı.")
    else:
        names = [s["ad"] for s in sepetler]
        T = pd.concat({s["ad"]: s["res"]["total"] for s in sepetler}, axis=1)
        rt = g["risk_table"](T, names[0], rf, alpha).astype(float)
        mets = []
        for s in sepetler:
            res, t = s["res"], s["res"]["table"]
            tot, net = t["Tutar (TL)"].sum(), t["Net değer (TL)"].sum()
            mets.append({"net": net / tot - 1, "brut": res["ret"], "vol": res["vol"], "dd": res["dd"],
                         "rv": g["risk_degeri"](res["total"])[0], "n": len(t),
                         "sh": rt.loc[s["ad"], "Sharpe"] if s["ad"] in rt.index else np.nan,
                         "yg": rt.loc[s["ad"], "Yıllık getiri"] if s["ad"] in rt.index else np.nan})
        c["kars_met"] = dict(zip(names, mets))

        st.markdown("##### Önerilen sepet ve örnek sepetler")
        cards = []
        for i, (s, m) in enumerate(zip(sepetler, mets)):
            tot_r = s["res"]["ret"]
            fon = "".join(f"<span class='ch'>{_h.escape(r['Kod'])} {r['Ağırlık'] * 100:.0f}%</span>"
                          for _, r in s["sec"].sort_values("Ağırlık", ascending=False).iterrows())
            sn = s["sec"].groupby("Sınıf")["Ağırlık"].sum().sort_values(ascending=False)
            sin = ", ".join(f"{k} {w * 100:.0f}%" for k, w in sn.items())
            uyar = ("<div class='bad'>⚠ Profilinizin riskinin üzerinde olabilir; yalnızca karşılaştırma içindir</div>"
                    if s.get("atak") else "")
            cards.append(
                f"<div class='bk {'me' if s.get('ana') else ''}'>"
                f"<div class='bt'>{_h.escape(s['ad'])}{'<span class=\"tag\">SİZİN SEPETİNİZ</span>' if s.get('ana') else ''}</div>"
                f"<div class='bs'>{_h.escape(s['aciklama'])}</div>{_spark(s['res']['total'], tot_r >= 0)}"
                f"<div class='chips2'>{_chip('Net', m['net'])}<span class='ch'>Vol {m['vol'] * 100:.1f}%</span>"
                f"<span class='ch'>Risk {'-' if pd.isna(m['rv']) else int(m['rv'])}/7</span></div>"
                f"<div class='bf'>{fon}</div><div class='wl' style='margin-top:6px'><span>{_h.escape(sin)}</span></div>{uyar}</div>")
        st.markdown("<div class='bkgrid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

        st.markdown("##### Parametre karşılaştırması")
        head = "".join(f"<th>{_h.escape(n)}</th>" for n in names)
        sat = [_bsk_sat(g, "Fon sayısı", [m["n"] for m in mets], lambda x: f"{int(x)}"),
               _bsk_sat(g, "Net getiri", [m["net"] for m in mets], pc, "high"),
               _bsk_sat(g, "Brüt dönem getirisi", [m["brut"] for m in mets], pc, "high"),
               _bsk_sat(g, "Yıllık getiri", [m["yg"] for m in mets], pc, "high"),
               _bsk_sat(g, "Volatilite", [m["vol"] for m in mets], pc, "low"),
               _bsk_sat(g, "Maks. düşüş", [m["dd"] for m in mets], pc, "high"),
               _bsk_sat(g, "Sharpe", [m["sh"] for m in mets], lambda x: f"{x:.2f}", "high"),
               _bsk_sat(g, "Risk değeri (1-7)", [m["rv"] for m in mets], lambda x: f"{int(x)}", "low")]
        st.markdown("<div class='tw'><table class='tbl cmp'><tr><th></th>" + head + "</tr>" + "".join(sat) +
                    "</table></div><div class='rnote'>Yeşil: ilgili satırdaki en iyi değer.</div>", unsafe_allow_html=True)

        st.markdown("##### Değer gelişimi (başlangıç = 100)")
        base = T / T.apply(lambda x: x.dropna().iloc[0]) * 100
        _ciz(lambda: g["line_chart"](base, th, "", ",.0f", 300))
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("##### Risk / getiri konumu")
            sc = (rt[["Yıllık volatilite", "Yıllık getiri"]].rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
                  .rename_axis("Fon").reset_index())
            _ciz(lambda: g["scatter_chart"](sc, th, ""))
        with k2:
            st.markdown("##### Net getiri")
            nd = pd.DataFrame({"Fon": names, "Getiri": [m["net"] for m in mets]})
            _ciz(lambda: g["bar_pos_neg"](nd, "Fon", "Getiri", "", th))
        st.markdown("##### Varlık sınıfı dağılımları")
        cols = st.columns(len(sepetler))
        for col, s in zip(cols, sepetler):
            with col:
                st.caption(s["ad"])
                dd = s["sec"].groupby("Sınıf")["Ağırlık"].sum().rename("Deger").rename_axis("Sınıf").reset_index()
                _ciz(lambda dd=dd: g["donut_chart"](dd, "Sınıf", "Deger", th, "", 200))

    st.markdown("##### Aynı tutar tek bir fonda değerlendirilseydi")
    _tek_fon(g, c)


def _tek_fon(g, c):
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
        st.markdown(f"##### Net getiri (her biri {g['tl'](c['tutar'])})")
        ad = a.rename_axis("Fon").reset_index()
        _ciz(lambda: g["bar_pos_neg"](ad, "Fon", "Getiri", "", th))
    with k2:
        st.markdown("##### Net kâr / zarar (TL)")
        bd = b.rename_axis("Fon").reset_index()
        _ciz(lambda: g["bar_pos_neg"](bd, "Fon", "Sonuc", "", th, ".2s"))


GORSEL = {"sepet": gor_sepet, "perf": gor_perf, "risk": gor_risk, "kars": gor_kars}
