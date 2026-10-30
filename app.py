"""TEFAS Fon Analiz Paneli v3 - tutarlı getiri/risk hesaplaması
Çalıştır: streamlit run app.py   (requirements.txt aynı kalabilir)
"""
import numpy as np
import pandas as pd
import streamlit as st
from etf_panel import (render_etf, render_etf_analiz, render_etf_detay,
                       etf_secenekleri, etf_tl_fiyat, etf_gider_orani)
from profil_panel import render_profil, render_oneri

st.set_page_config(page_title="Fon Analiz Paneli", layout="wide")
TD = 252


# ---------- Veri ----------
@st.cache_data(show_spinner="TEFAS verisi çekiliyor...", ttl=3600)
def load_tefas(codes, start, end):
    from tefas import Crawler  # pip install tefas-crawler
    crawler = Crawler()
    attempts = [
        ["code", "date", "price", "market_cap", "number_of_investors", "title"],
        ["code", "date", "price", "market_cap", "number_of_shares",
         "number_of_investors", "title"],
        ["code", "date", "price"],
    ]
    frames = []
    for code in codes:
        df = None
        for cols in attempts:
            try:
                df = crawler.fetch(start=start, end=end, name=code, columns=cols)
                break
            except Exception:
                continue
        if df is None:
            raise RuntimeError(f"{code} için TEFAS verisi alınamadı")
        frames.append(df)
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_synthetic(codes, start, end, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    frames = []
    for c in codes:
        mu, sg = rng.uniform(0.0003, 0.0012), rng.uniform(0.002, 0.012)
        r = rng.normal(mu, sg, len(idx))
        r[rng.integers(0, len(idx), 2)] -= 4 * sg  # anomali tohumu
        price = 100 * np.exp(np.cumsum(r))
        shares = 1e6 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(idx))))
        inv = (5000 * np.exp(np.cumsum(rng.normal(0.0005, 0.004, len(idx))))).round()
        frames.append(pd.DataFrame({
            "code": c, "date": idx, "price": price,
            "number_of_shares": shares, "market_cap": shares * price,
            "number_of_investors": inv, "title": f"{c} Demo Fon"}))
    return pd.concat(frames)


def to_wide(df):
    out = {}
    df = df.copy()
    for col in ["price", "market_cap", "number_of_shares", "number_of_investors"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            out[col] = df.pivot_table(index="date", columns="code", values=col,
                                      aggfunc="last").sort_index()
    # Fiyat serisi doldurulmaz: boşluklar sahte sıfır getiri üretmesin
    out["price"] = out["price"].dropna(how="all")
    if "number_of_shares" not in out and "market_cap" in out:
        out["number_of_shares"] = (out["market_cap"].ffill() / out["price"]) \
            .replace([np.inf, -np.inf], np.nan)
    return out


# ---------- Getiri ve risk çekirdeği ----------
def returns_df(p):
    """Her fon kendi gözlem tarihlerinde: r_t = P_t / P_(t-1) - 1."""
    return pd.concat({c: p[c].dropna().pct_change() for c in p}, axis=1)


def obs_per_year(s):
    """Yıllık gözlem sayısı = gözlem / (takvim günü / 365,25)."""
    s = s.dropna()
    days = (s.index[-1] - s.index[0]).days
    return (len(s) - 1) / (days / 365.25) if days > 0 and len(s) > 1 else float(TD)


def period_returns(p):
    last = p.index[-1]
    cut = {
        "1 Hafta": last - pd.Timedelta(days=7),
        "1 Ay": last - pd.DateOffset(months=1),
        "3 Ay": last - pd.DateOffset(months=3),
        "6 Ay": last - pd.DateOffset(months=6),
        "Yıl başı": pd.Timestamp(year=last.year - 1, month=12, day=31),
        "1 Yıl": last - pd.DateOffset(years=1),
    }
    rows = {}
    for c in p:
        s = p[c].dropna()
        row = {}
        for k, t in cut.items():
            v = s.asof(t)
            row[k] = s.iloc[-1] / v - 1 if pd.notna(v) else np.nan
        rows[c] = row
    df = pd.DataFrame(rows).T
    df = df.astype(float)
    df.insert(0, "Sıra (3 Ay)", df["3 Ay"].rank(ascending=False, method="min").astype("Int64"))
    return df.sort_values("Sıra (3 Ay)")


def risk_table(p, bench, rf=0.40, alpha=0.95):
    R = returns_df(p)
    rows = {}
    for c in p:
        s = p[c].dropna()
        if len(s) < 21:
            continue
        r = R[c].dropna()
        days = (s.index[-1] - s.index[0]).days
        ppy = obs_per_year(s)
        total = s.iloc[-1] / s.iloc[0] - 1
        ann = (1 + total) ** (365.25 / days) - 1
        vol = r.std() * np.sqrt(ppy)
        rf_d = (1 + rf) ** (1 / ppy) - 1          # risksiz oran, gözlem periyoduna
        ex = r - rf_d
        sharpe = ex.mean() / ex.std() * np.sqrt(ppy) if ex.std() else np.nan
        dd = (s / s.cummax() - 1).min()
        var = -np.percentile(r, (1 - alpha) * 100)
        cvar = -r[r <= -var].mean()
        row = {"Dönem getirisi": total, "Yıllık getiri": ann,
               "Yıllık volatilite": vol, "Sharpe": sharpe, "Maks. düşüş": dd,
               f"VaR %{int(alpha*100)} (1g)": var,
               f"CVaR %{int(alpha*100)} (1g)": cvar}
        if c != bench and bench in p:
            pair = p[[c, bench]].dropna().pct_change().dropna()  # ortak tarihler
            if len(pair) > 1:
                row["Tracking error"] = (pair[c] - pair[bench]).std() * np.sqrt(ppy)
        rows[c] = row
    return pd.DataFrame(rows).T








# ---------- Yatırımcı karşılaştırması ----------
PERIODS = {"1 Ay": pd.DateOffset(months=1), "3 Ay": pd.DateOffset(months=3),
           "6 Ay": pd.DateOffset(months=6), "1 Yıl": pd.DateOffset(years=1),
           "Tüm veri": None}


def tl(x):
    return f"{x:,.0f}".replace(",", ".") + " TL"


def pc(x):
    s = f"%{abs(x) * 100:.2f}".replace(".", ",")
    return ("-" + s) if x < 0 else s


def signed_tl(x):
    return ("+" if x >= 0 else "-") + tl(abs(x))


TR_MAP = str.maketrans("İıŞşĞğÜüÖöÇç", "IISSGGUUOOCC")
DEFAULT_STOPAJ = 0.175  # gerçek kişi, genel yatırım fonu (9 Temmuz 2025 kararı); değişebilir


def auto_stopaj(title):
    """Fon adından tahmini stopaj oranı. Kesin sınıflandırma için KAP/izahname kontrol edilmeli."""
    t = str(title).translate(TR_MAP).upper()
    if "HISSE SENEDI YOGUN" in t or "HISSE YOGUN" in t:
        return 0.0, "hisse senedi yoğun"
    if "GAYRIMENKUL YATIRIM" in t or "GIRISIM SERMAYESI" in t:
        return DEFAULT_STOPAJ, "GYF/GSYF (2 yıldan uzun elde tutulursa %0)"
    return DEFAULT_STOPAJ, "genel oran"


def investor_metrics(p, funds, amount, offset, rf, rates, buy_fee=0.0, sell_fee=0.0):
    """Tutarın tamamı her fonda değerlendirilseydi: brüt, stopaj, masraf, net sonuç."""
    last = p.index[-1]
    rows = {}
    for c in funds:
        s = p[c].dropna()
        if len(s) < 5:
            continue
        if offset is None:
            seg = s
        else:
            before = s.index[s.index <= last - offset]
            seg = s[s.index >= before[-1]] if len(before) else s
        days = (seg.index[-1] - seg.index[0]).days
        if days < 5:
            continue
        r = seg.pct_change().dropna()
        total = seg.iloc[-1] / seg.iloc[0] - 1
        ppy = obs_per_year(seg)
        enough = len(r) >= 10
        vol = r.std() * np.sqrt(ppy) if enough else np.nan
        rf_d = (1 + rf) ** (1 / ppy) - 1
        ex = r - rf_d
        sharpe = ex.mean() / ex.std() * np.sqrt(ppy) if enough and ex.std() > 0 else np.nan
        dd = (seg / seg.cummax() - 1).min()
        roll = seg.pct_change(21).dropna()          # ~1 aylık (21 gözlem) getiriler
        p5 = np.percentile(roll, 5) if len(roll) >= 20 else np.nan
        # Vergi ve masraf: stopaj yalnızca kazançtan, satışta kesilir
        net_in = amount * (1 - buy_fee)
        gross_val = net_in * (1 + total)
        gross_gain = gross_val - net_in
        rate = rates.get(c, DEFAULT_STOPAJ)
        stopaj = max(gross_gain, 0.0) * rate
        cost = amount * buy_fee + gross_val * sell_fee
        net_val = gross_val - stopaj - gross_val * sell_fee
        rows[c] = {
            "Brüt fon getirisi": total,
            "Brüt dönem sonu değer (TL)": gross_val,
            "Brüt kâr/zarar (TL)": gross_gain,
            "Stopaj oranı": rate,
            "Stopaj (TL)": stopaj,
            "Masraf (TL)": cost,
            "Net elde edilen (TL)": net_val,
            "Net kâr/zarar (TL)": net_val - amount,
            "Net getiri": net_val / amount - 1,
            "Yıllık volatilite": vol,
            "Maks. düşüş": dd,
            "Maks. düşüş (TL)": amount * dd,
            "Kötü ay %5 (getiri)": p5,
            "Kötü ay %5 (TL)": amount * p5 if pd.notna(p5) else np.nan,
            "Sharpe": sharpe,
            "Başlangıç tarihi": seg.index[0].strftime("%d.%m.%Y"),
            "Gün": days,
        }
    return pd.DataFrame(rows).T


def investor_text(m, amount, label, rf):
    m = m.dropna(subset=["Net getiri"])
    if len(m) < 2:
        return ["Karşılaştırma için en az iki fonda yeterli veri yok."]
    out = []
    top, low = m["Net getiri"].idxmax(), m["Net getiri"].idxmin()
    g = m.loc[top]
    out.append(
        f"**Net getiri (stopaj ve masraf sonrası):** Geçmişte {label} içinde {tl(amount)} "
        f"yalnızca **{top}** fonunda değerlendirilmiş olsaydı: brüt kâr/zarar "
        f"{signed_tl(g['Brüt kâr/zarar (TL)'])}, stopaj {tl(g['Stopaj (TL)'])} "
        f"({pc(g['Stopaj oranı'])}), masraf {tl(g['Masraf (TL)'])}; elinize geçen yaklaşık "
        f"**{tl(g['Net elde edilen (TL)'])}** (net {pc(g['Net getiri'])}, "
        f"{signed_tl(g['Net kâr/zarar (TL)'])}). En düşük net getiri **{low}** fonunda "
        f"olurdu: {pc(m.loc[low, 'Net getiri'])} "
        f"({signed_tl(m.loc[low, 'Net kâr/zarar (TL)'])}).")
    top_gross = m["Brüt fon getirisi"].idxmax()
    if top_gross != top:
        out.append(
            f"Brüt getiride ilk sıradaki fon **{top_gross}** ({pc(m.loc[top_gross, 'Brüt fon getirisi'])}) "
            f"iken, stopaj ve masraflar sonrasında ilk sıra **{top}** fonuna geçmiştir.")
    parts = [f"{c}: stopaj {pc(m.loc[c, 'Stopaj oranı'])} = {tl(m.loc[c, 'Stopaj (TL)'])}, "
             f"masraf {tl(m.loc[c, 'Masraf (TL)'])}" for c in m.index]
    out.append("**Stopaj ve masraf özeti:** " + "; ".join(parts) + ".")
    lv = m["Yıllık volatilite"].dropna()
    dd = m["Maks. düşüş"].dropna()
    safe = lv.idxmin() if len(lv) else None
    if safe is not None:
        risky = lv.idxmax()
        txt = (f"**Risk:** Fiyat dalgalanması en düşük fon **{safe}** "
               f"(yıllık volatilite {pc(lv[safe])}), en yüksek fon **{risky}** "
               f"({pc(lv[risky])}).")
        if len(dd):
            sdd, rdd = dd.idxmax(), dd.idxmin()
            txt += (f" Dönem içindeki en büyük değer kaybı **{sdd}** fonunda en az "
                    f"({pc(dd[sdd])}, yaklaşık {tl(abs(m.loc[sdd, 'Maks. düşüş (TL)']))}), "
                    f"**{rdd}** fonunda en fazla olmuştur ({pc(dd[rdd])}, yaklaşık "
                    f"{tl(abs(m.loc[rdd, 'Maks. düşüş (TL)']))}).")
        out.append(txt)
        if top == safe:
            out.append(f"**{top}** fonu bu dönemde hem en yüksek net getiriyi sağlamış hem de "
                       "en düşük dalgalanmayı göstermiştir.")
        else:
            out.append(
                f"Bu dönemde en yüksek net getiri ile en düşük risk aynı fonda değildir: "
                f"daha yüksek net getiri sağlayan **{top}** fonunun volatilitesi "
                f"{pc(m.loc[top, 'Yıllık volatilite'])}, daha düşük riskli **{safe}** "
                f"fonunun net getirisi {pc(m.loc[safe, 'Net getiri'])} olmuştur.")
    sh = m["Sharpe"].dropna()
    if len(sh):
        best = sh.idxmax()
        txt = (f"**Risk başına getiri (Sharpe, brüt):** en yüksek değer **{best}** fonunda "
               f"({sh[best]:.2f}).")
        if sh.max() < 0:
            txt += (f" Hiçbir fonun getirisi {pc(rf)} yıllık risksiz oran varsayımını "
                    "aşamamıştır, bu nedenle Sharpe değerleri negatiftir.")
        out.append(txt)
    out.append("Bu değerlendirme yalnızca geçmiş verilere dayanır; gelecekteki getiri veya "
               "riskin göstergesi ya da garantisi değildir. Stopaj, satışta kazanç üzerinden "
               "kesilir ve oranlar fon adından tahmin edilmiştir; kesin oran fonun izahnamesi "
               "ve güncel mevzuattan teyit edilmelidir. Fon yönetim ücreti fon fiyatına "
               "dahildir. Enflasyon etkisi dahil değildir. Yatırım tavsiyesi niteliği taşımaz.")
    return out


# ---------- Tema ve görünüm (açık, kurumsal) ----------
from string import Template

THEMES = {
    "Açık": dict(
        bg="#ffffff", card="#ffffff", card2="#eff4ff", sidebar="#f8fafc", border="#e5e7eb",
        text="#111827", muted="#6b7280", accent="#1d4ed8", up="#15803d", down="#b91c1c",
        upbg="#dcfce7", downbg="#fee2e2", flatbg="#f3f4f6",
        palette=["#1d4ed8", "#0d9488", "#d97706", "#7c3aed", "#dc2626", "#0891b2",
                 "#4b5563", "#be185d"],
        font="'Inter','Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif",
        chartfont="Inter", radius="10px", cr=3,
        hdrbg="linear-gradient(135deg,#f8fafc 0%,#eef2f7 100%)"),
}

THEMES["Koyu"] = dict(
    bg="#0a1633", card="#10214a", card2="#172b5c", sidebar="#081229", border="#24396b",
    text="#ffffff", muted="#a9b8d9", accent="#60a5fa", up="#34d399", down="#f87171",
    upbg="rgba(52,211,153,0.15)", downbg="rgba(248,113,113,0.15)", flatbg="#1a2d5a",
    palette=["#60a5fa", "#2dd4bf", "#fbbf24", "#a78bfa", "#f472b6", "#34d399",
             "#fb923c", "#22d3ee"],
    font="'Inter','Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif",
    chartfont="Inter", radius="10px", cr=3,
    hdrbg="linear-gradient(135deg,#10214a 0%,#0a1633 100%)")

CSS_TMPL = Template("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp {background:$bg; color:$text; font-family:$font;}
.block-container {padding-top:1.4rem; max-width:1250px;}
h1, h2, h3, h4, h5, h6 {color:$text !important; font-family:$font !important; font-weight:600;}
h5 {color:$muted !important; font-size:0.76rem !important; text-transform:uppercase;
  letter-spacing:1.1px; margin-top:0.8rem; margin-bottom:0.2rem;}
.stApp p, .stApp label, .stMarkdown {color:$text;}
div[data-testid="stCaptionContainer"] {color:$muted !important;}
section[data-testid="stSidebar"] {background:$sidebar; border-right:1px solid $border;}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {background:$card !important;
  border-color:$border !important;}
div[data-baseweb="input"] input, div[data-baseweb="select"] div {color:$text !important;}
div[data-testid="stExpander"] {border:1px solid $border; border-radius:$radius;}
button[data-baseweb="tab"] {color:$muted; border-radius:999px; padding:6px 14px; margin-right:4px;}
button[data-baseweb="tab"][aria-selected="true"] {color:$accent; background:$card2;}
div[data-baseweb="tab-highlight"] {background:$accent !important;}
div[data-baseweb="tab-border"] {background:$border !important;}
div[data-testid="stMetric"] {background:$card; border:1px solid $border; border-radius:$radius;
  padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,0.06);}
div[data-testid="stMetricLabel"] p {color:$muted; font-size:0.72rem; text-transform:uppercase;
  letter-spacing:1px;}
div[data-testid="stMetricValue"] {color:$text; font-weight:700;}
.hdr {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
  background:$hdrbg; border:1px solid $border; border-radius:$radius; padding:16px 20px;
  margin-bottom:14px;}
.brand {font-size:22px; font-weight:700; color:$text;}
.sub {color:$muted; font-size:13px; margin-top:2px;}
.date {color:$muted; font-size:12px; margin-left:8px;}
.pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px;
  font-weight:600; letter-spacing:0.6px;}
.pill.live {background:$upbg; color:$up;}
.pill.demo {background:$flatbg; color:$muted;}
.tk-wrap {display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px;
  margin-bottom:14px;}
.tk {background:$card; border:1px solid $border; border-top:3px solid $accent;
  border-radius:$radius; padding:12px 14px; box-shadow:0 1px 2px rgba(16,24,40,0.06);}
.tk.up {border-top-color:$up;} .tk.down {border-top-color:$down;}
.tk .code {color:$muted; font-size:12px; font-weight:600; letter-spacing:1px;}
.tk .px {color:$text; font-size:20px; font-weight:700; font-variant-numeric:tabular-nums;
  margin:2px 0 8px;}
.chips {display:flex; gap:6px; flex-wrap:wrap;}
.chip {font-size:11px; padding:2px 8px; border-radius:999px; font-weight:600;}
.chip.up {background:$upbg; color:$up;}
.chip.down {background:$downbg; color:$down;}
.chip.flat {background:$flatbg; color:$muted;}
""")


def make_css(th):
    return "<style>" + CSS_TMPL.substitute(th) + "</style>"


def pal(th, n):
    return [th["palette"][i % len(th["palette"])] for i in range(n)]


def _chip(label, x):
    if x is None or pd.isna(x):
        return f"<span class='chip flat'>{label} -</span>"
    cls = "up" if x >= 0 else "down"
    arrow = "▲" if x >= 0 else "▼"
    return f"<span class='chip {cls}'>{label} {arrow}{abs(x) * 100:.2f}%</span>"


def ticker_html(p):
    """Her fon için son fiyat ve 1G / 1A / YBB değişim kartları."""
    last = p.index[-1]
    ytd_ref = pd.Timestamp(year=last.year - 1, month=12, day=31)
    cards = []
    for c in p:
        s = p[c].dropna()
        if len(s) < 2:
            continue
        d1 = s.iloc[-1] / s.iloc[-2] - 1
        m_ref = s.asof(last - pd.DateOffset(months=1))
        m1 = s.iloc[-1] / m_ref - 1 if pd.notna(m_ref) else np.nan
        y_ref = s.asof(ytd_ref)
        ytd = s.iloc[-1] / y_ref - 1 if pd.notna(y_ref) else np.nan
        direction = "up" if d1 >= 0 else "down"
        cards.append(
            f"<div class='tk {direction}'><div class='code'>{c}</div>"
            f"<div class='px'>{s.iloc[-1]:,.4f}</div>"
            f"<div class='chips'>{_chip('1G', d1)}{_chip('1A', m1)}{_chip('YBB', ytd)}</div></div>")
    return "<div class='tk-wrap'>" + "".join(cards) + "</div>"


# ---------- Grafikler (Altair) ----------
def _style_alt(ch, title, th, height=230):
    ch = ch.properties(title=title, height=height, background=th["bg"])
    return (ch.configure_axis(labelColor=th["muted"], titleColor=th["muted"],
                              gridColor=th["border"], domain=False, ticks=False,
                              labelFont=th["chartfont"], titleFont=th["chartfont"],
                              labelFontSize=11)
              .configure_axisX(grid=False)
              .configure_view(stroke=None)
              .configure_title(color=th["text"], anchor="start", font=th["chartfont"],
                               fontSize=13, fontWeight=600)
              .configure_legend(labelColor=th["text"], titleColor=th["muted"],
                                labelFont=th["chartfont"], labelFontSize=11))


def line_chart(df, th, title="", fmt=",.0f", height=230):
    """Çok serili çizgi grafik; df: tarih indeksli, sütunlar = seri."""
    import altair as alt
    long = (df.rename_axis("Tarih").reset_index()
              .melt(id_vars="Tarih", var_name="Fon", value_name="Deger").dropna())
    order = list(df.columns)
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("Tarih:T", title=None),
        y=alt.Y("Deger:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=fmt)),
        color=alt.Color("Fon:N", sort=order,
                        scale=alt.Scale(domain=order, range=pal(th, len(order))),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Tarih:T", format="%d.%m.%Y"), "Fon:N",
                 alt.Tooltip("Deger:Q", format=fmt)])
    return _style_alt(ch, title, th, height)


def stacked_area(df, th, title="", fmt=",.0f", height=230):
    """Yığılmış alan grafiği (üst çizgi toplamı verir)."""
    import altair as alt
    long = (df.rename_axis("Tarih").reset_index()
              .melt(id_vars="Tarih", var_name="Fon", value_name="Deger").dropna())
    order = list(df.columns)
    ch = alt.Chart(long).mark_area(opacity=0.9).encode(
        x=alt.X("Tarih:T", title=None),
        y=alt.Y("Deger:Q", stack=True, title=None, axis=alt.Axis(format=fmt)),
        color=alt.Color("Fon:N", sort=order,
                        scale=alt.Scale(domain=order, range=pal(th, len(order))),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Tarih:T", format="%d.%m.%Y"), "Fon:N",
                 alt.Tooltip("Deger:Q", format=fmt)])
    return _style_alt(ch, title, th, height)


def bar_pos_neg(df, x, y, title, th, fmt=".1%", height=230):
    """Pozitif yeşil, negatif kırmızı çubuklar; değer etiketli."""
    import altair as alt
    order = list(df.sort_values(y, ascending=False)[x])
    base = alt.Chart(df).encode(x=alt.X(f"{x}:N", sort=order, title=None,
                                        axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(cornerRadiusEnd=th["cr"], size=28).encode(
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(format=fmt)),
        color=alt.condition(f"datum.{y} >= 0", alt.value(th["up"]), alt.value(th["down"])),
        tooltip=[x, alt.Tooltip(f"{y}:Q", format=fmt)])
    pos = (base.mark_text(dy=-8, fontSize=11, color=th["text"])
               .encode(y=alt.Y(f"{y}:Q"), text=alt.Text(f"{y}:Q", format=fmt))
               .transform_filter(f"datum.{y} >= 0"))
    neg = (base.mark_text(dy=10, fontSize=11, color=th["text"])
               .encode(y=alt.Y(f"{y}:Q"), text=alt.Text(f"{y}:Q", format=fmt))
               .transform_filter(f"datum.{y} < 0"))
    return _style_alt(bars + pos + neg, title, th, height)


def scatter_chart(df, th, title="Risk / Getiri", height=230):
    """df sütunları: Fon, Vol, Getiri."""
    import altair as alt
    base = alt.Chart(df).encode(
        x=alt.X("Vol:Q", title="Yıllık volatilite", axis=alt.Axis(format=".0%")),
        y=alt.Y("Getiri:Q", title="Yıllık getiri", axis=alt.Axis(format=".0%")))
    pts = base.mark_circle(size=140, color=th["accent"], opacity=0.9).encode(
        tooltip=["Fon", alt.Tooltip("Vol:Q", format=".2%"),
                 alt.Tooltip("Getiri:Q", format=".2%")])
    txt = base.mark_text(dy=-14, color=th["text"], font=th["chartfont"],
                         fontSize=11).encode(text="Fon:N")
    return _style_alt(pts + txt, title, th, height)


def donut_chart(df, label, value, th, title="", height=270):
    """Daire (halka) grafik; oranlar dilim üzerinde gösterilir."""
    import altair as alt
    d = df[[label, value]].copy()
    d = d[d[value] > 0]
    d["Oran"] = d[value] / d[value].sum()
    order = list(d.sort_values(value, ascending=False)[label])
    base = alt.Chart(d).encode(
        theta=alt.Theta(f"{value}:Q", stack=True),
        order=alt.Order(f"{value}:Q", sort="descending"),
        color=alt.Color(f"{label}:N", sort=order,
                        scale=alt.Scale(domain=order, range=pal(th, len(order))),
                        legend=alt.Legend(title=None, orient="bottom")))
    arcs = base.mark_arc(innerRadius=56, outerRadius=98, stroke="#ffffff",
                         strokeWidth=2).encode(
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=",.0f"),
                 alt.Tooltip("Oran:Q", format=".1%")])
    txt = (base.mark_text(radius=116, size=11, color=th["text"])
               .encode(text=alt.Text("Oran:Q", format=".0%"))
               .transform_filter("datum.Oran >= 0.04"))
    return _style_alt(arcs + txt, title, th, height)


def corr_heatmap(corr, th, title=""):
    """Korelasyon ısı haritası (kırmızı negatif, mavi pozitif)."""
    import altair as alt
    order = list(corr.columns)
    cl = (corr.rename_axis("A").reset_index()
              .melt(id_vars="A", var_name="B", value_name="Kor"))
    base = alt.Chart(cl).encode(
        x=alt.X("B:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("A:N", sort=order, title=None))
    rect = base.mark_rect(stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color("Kor:Q", legend=None,
                        scale=alt.Scale(domain=[-1, 0, 1],
                                        range=["#b91c1c", "#f8fafc", "#1d4ed8"])),
        tooltip=["A", "B", alt.Tooltip("Kor:Q", format=".2f")])
    txt = base.mark_text(fontSize=12).encode(
        text=alt.Text("Kor:Q", format=".2f"),
        color=alt.condition("abs(datum.Kor) > 0.6", alt.value("#ffffff"),
                            alt.value("#111827")))
    return _style_alt(rect + txt, title, th, height=max(170, 46 * len(order) + 40))


# ---------- Portföy sepeti ----------
def basket_analysis(p, amounts, offset, rates, buy_fee=0.0, sell_fee=0.0, alpha=0.95,
                    annual_er=None):
    """Fon başına tutar girilen sepetin (al-tut, yeniden dengeleme yok) analizi."""
    funds = [c for c, a in amounts.items() if a and a > 0 and c in p.columns]
    if not funds:
        return None
    sub = p[funds].dropna()                      # ortak tarihler
    if offset is not None and len(sub):
        before = sub.index[sub.index <= sub.index[-1] - offset]
        if len(before):
            sub = sub[sub.index >= before[-1]]
    if len(sub) < 10:
        return None
    amt = pd.Series({c: float(amounts[c]) for c in funds})
    w = amt / amt.sum()
    value = sub.mul(amt / sub.iloc[0], axis=1)   # her fonun TL değeri
    total = value.sum(axis=1)
    r = total.pct_change().dropna()
    days = (sub.index[-1] - sub.index[0]).days
    ppy = (len(sub) - 1) / (days / 365.25) if days > 0 else float(TD)
    vol = r.std() * np.sqrt(ppy)
    dd = (total / total.cummax() - 1).min()
    var = -np.percentile(r, (1 - alpha) * 100)
    cvar = -r[r <= -var].mean()
    cov = sub.pct_change().dropna().cov().values * ppy
    wv = w.values
    pv = float(wv @ cov @ wv)
    if pv > 0:
        rc = (wv * (cov @ wv)) / pv              # risk katkısı, toplamı 1
        div = float((wv * np.sqrt(np.diag(cov))).sum() / np.sqrt(pv))
    else:
        rc, div = np.full(len(funds), np.nan), np.nan
    fret = sub.iloc[-1] / sub.iloc[0] - 1
    bf = pd.Series({c: (buy_fee.get(c, 0.0) if isinstance(buy_fee, dict) else buy_fee)
                    for c in funds})
    sf = pd.Series({c: (sell_fee.get(c, 0.0) if isinstance(sell_fee, dict) else sell_fee)
                    for c in funds})
    er = pd.Series({c: float((annual_er or {}).get(c, 0.0)) for c in funds})
    net_in = amt * (1 - bf)
    gross0 = net_in * (1 + fret)
    exp_tl = gross0 * (1 - (1 - er) ** (days / 365.25))   # ETF gider oranı, süreye orantılı
    gross = gross0 - exp_tl
    gain = gross - net_in
    rate = pd.Series({c: rates.get(c, DEFAULT_STOPAJ) for c in funds})
    stopaj = gain.clip(lower=0) * rate           # stopaj yalnızca kazançtan, fon bazında
    sell = gross * sf
    net_val = gross - stopaj - sell
    table = pd.DataFrame({
        "Tutar (TL)": amt, "Ağırlık": w, "Dönem getirisi": fret,
        "Risk katkısı": pd.Series(rc, index=funds),
        "Brüt değer (TL)": gross, "Stopaj (TL)": stopaj,
        "Masraf (TL)": amt * bf + sell + exp_tl, "Gider oranı (TL)": exp_tl,
        "Net değer (TL)": net_val,
        "Net kâr/zarar (TL)": net_val - amt})
    return {"table": table, "value": value, "total": total,
            "ret": total.iloc[-1] / total.iloc[0] - 1, "vol": vol, "dd": dd,
            "var": var, "cvar": cvar, "div": div, "start": sub.index[0],
            "alpha": alpha, "er": er, "days": days}


def basket_text(res, label):
    t = res["table"]
    tot = t["Tutar (TL)"].sum()
    out = []
    parts = [f"**{c}** {pc(t.loc[c, 'Ağırlık'])} ({tl(t.loc[c, 'Tutar (TL)'])})"
             for c in t.sort_values("Tutar (TL)", ascending=False).index]
    out.append(f"**Sepet:** Toplam {tl(tot)} şöyle dağılmıştır: " + ", ".join(parts) + ".")
    net_total = t["Net değer (TL)"].sum()
    out.append(
        f"**Getiri:** Geçmişte {label} içinde (ortak veri başlangıcı "
        f"{res['start']:%d.%m.%Y}) bu sepetin brüt getirisi {pc(res['ret'])} olurdu. "
        f"Stopaj {tl(t['Stopaj (TL)'].sum())} ve masraf {tl(t['Masraf (TL)'].sum())} "
        f"sonrasında elinize geçen yaklaşık **{tl(net_total)}** "
        f"({pc(net_total / tot - 1)}, {signed_tl(net_total - tot)}).")
    er = res.get("er")
    if er is not None and (er > 0).any():
        parts_e = [f"{c} yıllık %{er[c] * 100:.2f} (bu dönem yaklaşık "
                   f"{tl(t.loc[c, 'Gider oranı (TL)'])})" for c in er.index if er[c] > 0]
        out.append("**ETF gider oranı:** Sepetteki ETF'lerin gider oranı, dönem süresine "
                   "orantılı olarak getiriden otomatik düşülmüştür: " + ", ".join(parts_e) +
                   ". ETF'ler USD ile işlem gördüğünden getiri, USD/TRY kuruyla TL'ye "
                   "çevrilmiştir ve kur etkisini içerir. Not: Yahoo fiyatları gider oranını "
                   "zaten yansıttığı için bu kalem çift sayım olabilir; istemezseniz ETF "
                   "masraf ayarlarındaki kutuyu kapatın.")
    out.append(
        f"**Risk:** Sepetin yıllık volatilitesi {pc(res['vol'])}; dönem içindeki en büyük "
        f"değer kaybı {pc(res['dd'])} (yaklaşık {tl(abs(res['dd']) * tot)}). Tarihsel "
        f"verilerde günlük kayıp, %{int(res['alpha'] * 100)} güven düzeyinde "
        f"{pc(res['var'])} ({tl(res['var'] * tot)}) seviyesinin altında kalmıştır.")
    if pd.notna(res["div"]):
        if res["div"] > 1.05:
            out.append(f"**Çeşitlendirme:** Oran {res['div']:.2f}. Sepet riski, fonların "
                       f"ağırlıklı ortalama riskinden yaklaşık %{(1 - 1 / res['div']) * 100:.0f} "
                       "daha düşüktür; fonlar birbirini kısmen dengeliyor.")
        else:
            out.append(f"**Çeşitlendirme:** Oran {res['div']:.2f}; fonlar benzer yönde "
                       "hareket ettiği için çeşitlendirme faydası sınırlıdır.")
    rc = t["Risk katkısı"].dropna()
    if len(rc) > 1:
        top = rc.idxmax()
        if rc[top] > t.loc[top, "Ağırlık"] + 0.05:
            out.append(f"**Risk yoğunlaşması:** **{top}** fonu tutarın {pc(t.loc[top, 'Ağırlık'])}"
                       f"'ini oluşturuyor, ancak sepet riskinin {pc(rc[top])}'ini taşıyor.")
        else:
            out.append("**Risk dağılımı:** Fonların risk payları tutar paylarıyla büyük "
                       "ölçüde uyumludur.")
    pr = t["Net kâr/zarar (TL)"]
    if len(pr) > 1:
        out.append(f"**Katkı:** Net sonuca en çok katkı veren fon **{pr.idxmax()}** "
                   f"({signed_tl(pr.max())}), en az katkı veren **{pr.idxmin()}** "
                   f"({signed_tl(pr.min())}).")
    out.append("Değerlendirme geçmiş verilere dayanır, gelecekteki getiri veya riskin "
               "garantisi değildir; yatırım tavsiyesi niteliği taşımaz. Sepet, al-tut "
               "varsayımıyla (yeniden dengeleme olmadan) hesaplanmıştır. Stopaj oranlarını "
               "izahnameden teyit ediniz.")
    return out


# ---------- Haftalık fon bülteni ----------






















# ---------- Performans tabloları ----------
def relative_table(p, bench):
    """Her dönemde fon getirisi - kıyas getirisi (yüzde puan)."""
    if bench not in p.columns:
        return None
    pr = period_returns(p).drop(columns=["Sıra (3 Ay)"], errors="ignore")
    if bench not in pr.index:
        return None
    return pr.sub(pr.loc[bench], axis=1).drop(index=bench, errors="ignore").astype(float)


def monthly_returns(p):
    try:
        m = p.resample("ME").last()
    except Exception:
        m = p.resample("M").last()
    r = m.pct_change().dropna(how="all")
    r.index = r.index.strftime("%Y-%m")
    return r.astype(float)


def monthly_heatmap(mr, th):
    import altair as alt
    vals = np.abs(mr.values.astype(float))
    vals = vals[~np.isnan(vals)]
    m = float(vals.max()) if vals.size else 0.01
    m = max(m, 1e-4)
    long = (mr.rename_axis("Ay").reset_index()
              .melt(id_vars="Ay", var_name="Fon", value_name="Getiri").dropna())
    order = list(mr.columns)
    base = alt.Chart(long).encode(
        x=alt.X("Ay:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Fon:N", sort=order, title=None))
    rect = base.mark_rect(stroke=th["bg"], strokeWidth=1).encode(
        color=alt.Color("Getiri:Q", legend=None,
                        scale=alt.Scale(domain=[-m, 0, m], range=[th["down"], th["card"], th["up"]],
                                        clamp=True)),
        tooltip=["Fon", "Ay", alt.Tooltip("Getiri:Q", format=".2%")])
    txt = base.mark_text(fontSize=10).encode(text=alt.Text("Getiri:Q", format=".1%"),
                                             color=alt.value(th["text"]))
    return _style_alt(rect + txt, "", th, height=max(160, 34 * len(order) + 60))




# ---------- Piyasalar (Yahoo Finance) ----------
# Yahoo'nun ABD faiz endeksleri değeri yüzde × 10 olarak verir (ör. 42,5 = %4,25)










# ---------- Varlık dağılımı ----------
ASSET_CLASSES = ["Hisse senedi", "Devlet tahvili/bonosu", "Özel sektör borçlanma",
                 "Repo ve mevduat", "Altın ve kıymetli maden", "Yabancı varlık", "Diğer"]
BREAK_LABEL = {"stock_pct": "Hisse senedi", "repo_pct": "Repo",
               "foreign_stock_pct": "Yabancı hisse", "precious_metals_pct": "Kıymetli maden"}


@st.cache_data(ttl=21600, show_spinner="TEFAS'tan portföy dağılımı çekiliyor...")
def load_breakdown(end_str):
    from pytefas import Crawler  # pip install pytefas
    tf = Crawler()
    last = pd.Timestamp(end_str)
    for back in range(7):
        d = (last - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        df = tf.fetch(start=d, columns="breakdown", kind="YAT")
        if df is not None and len(df):
            return df
    raise RuntimeError("Son 7 günde TEFAS portföy dağılımı verisi bulunamadı")


def breakdown_table(df, funds):
    """Seçilen fonların dağılım tablosu (yüzde). Toplam %100 olması gerekmez."""
    code = "fund_code" if "fund_code" in df.columns else None
    if code is None:
        raise RuntimeError("Dağılım verisinde fon kodu sütunu yok")
    pct_cols = [c for c in df.columns if str(c).endswith("_pct")]
    d = df[df[code].isin(funds)].drop_duplicates(code).set_index(code)
    out = d[pct_cols].apply(pd.to_numeric, errors="coerce")
    # Değerler oran olarak gelmişse (tüm değerler ≤ 1 ve satır toplamları ≈ 1) yüzdeye çevir
    if out.size and out.max().max() <= 1.0 and out.sum(axis=1).max() <= 1.05:
        out = out * 100
    out = out.loc[:, (out.fillna(0) != 0).any()]
    out.columns = [BREAK_LABEL.get(c, c.replace("_pct", "").replace("_", " ").capitalize())
                   for c in out.columns]
    return out


def alloc_bar(tbl, th):
    import altair as alt
    cats = list(tbl.columns)
    long = (tbl.rename_axis("Fon").reset_index()
              .melt(id_vars="Fon", var_name="Varlık", value_name="Oran").dropna())
    long = long[long["Oran"] > 0]
    ch = alt.Chart(long).mark_bar(size=44).encode(
        x=alt.X("Fon:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Oran:Q", stack="zero", title=None, axis=alt.Axis(format=".0f")),
        color=alt.Color("Varlık:N", sort=cats,
                        scale=alt.Scale(domain=cats, range=pal(th, len(cats))),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["Fon", "Varlık", alt.Tooltip("Oran:Q", format=".1f")])
    return _style_alt(ch, "", th, height=300)


# ---------- Fon sayıları ve pazar payı (TEFAS fon bilgisi) ----------
FUND_CAT_RULES = [
    ("Para piyasası", ["PARA PIYASASI", "LIKIT FON"]),
    ("GYF / GSYF", ["GAYRIMENKUL", "GSYF", "GYF", "GIRISIM"]),
    ("Altın ve kıymetli maden", ["ALTIN", "KIYMETLI"]),
    ("Yabancı / eurobond", ["YABANCI", "EUROBOND", "YURTDISI"]),
    ("Hisse senedi", ["HISSE"]),
    ("Borçlanma araçları", ["BORCLANMA", "TAHVIL", "BONO", "DEGISKEN", "KISA VADELI"]),
    ("Karma / çoklu varlık", ["COKLU", "KARMA", "SERBEST", "DENGELI", "SEPET"]),
]


def fund_category(name):
    """Fon adından kategori. Yaklaşık sınıflandırma: TEFAS tür alanı bu veride yok."""
    n = str(name).translate(TR_MAP).upper()
    for cat, keys in FUND_CAT_RULES:
        if any(k in n for k in keys):
            return cat
    return "Diğer"


def _num(s):
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(ttl=21600, show_spinner="TEFAS fon bilgileri çekiliyor...")
def load_fund_info(end_str, ref_days=30):
    from pytefas import Crawler  # pip install pytefas
    tf = Crawler()

    def get(ref):
        for back in range(7):
            d = (ref - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            df = tf.fetch_many(start=d, kinds=("YAT", "EMK", "BYF"), columns="info")
            if df is not None and len(df):
                return df, d
        return None, None

    last = pd.Timestamp(end_str)
    cur, cur_d = get(last)
    if cur is None:
        raise RuntimeError("Son 7 günde TEFAS fon bilgisi bulunamadı")
    ref, ref_d = get(pd.Timestamp(cur_d) - pd.Timedelta(days=ref_days))
    return cur, ref, cur_d, ref_d


def fund_table(cur, ref):
    c = cur.drop_duplicates("fund_code").set_index("fund_code")
    out = pd.DataFrame(index=c.index)
    out["Fon adı"] = c["fund_name"].astype(str)
    out["Tür"] = c["kind"].astype(str) if "kind" in c.columns else ""
    out["Kategori"] = [fund_category(n) for n in c["fund_name"]]
    out["Portföy büyüklüğü (TL)"] = _num(c["portfolio_size"])
    out["Yatırımcı sayısı"] = _num(c["investor_count"])
    if ref is not None and len(ref):
        r = ref.drop_duplicates("fund_code").set_index("fund_code").reindex(out.index)
        out["Önceki büyüklük (TL)"] = _num(r["portfolio_size"])
        out["Önceki yatırımcı"] = _num(r["investor_count"])
        out["Yatırımcı değişimi"] = out["Yatırımcı sayısı"] - out["Önceki yatırımcı"]
        out["Büyüklük değişimi (%)"] = (out["Portföy büyüklüğü (TL)"] /
                                        out["Önceki büyüklük (TL)"] - 1).where(
            out["Önceki büyüklük (TL)"] > 0)
        dsh = _num(c["shares_outstanding"]) - _num(r["shares_outstanding"])
        out["Net akış (TL, tahmini)"] = dsh * _num(c["price"])
    size, inv = out["Portföy büyüklüğü (TL)"], out["Yatırımcı sayısı"]
    out["Pazar payı (%)"] = size / size.sum()
    out["Yatırımcı payı (%)"] = inv / inv.sum()
    return out.replace([np.inf, -np.inf], np.nan)


def category_table(ft):
    agg = {"Fon sayısı": ("Fon adı", "count"),
           "Portföy büyüklüğü (TL)": ("Portföy büyüklüğü (TL)", "sum"),
           "Yatırımcı sayısı": ("Yatırımcı sayısı", "sum")}
    if "Önceki büyüklük (TL)" in ft.columns:
        agg["Önceki büyüklük (TL)"] = ("Önceki büyüklük (TL)", "sum")
        agg["Önceki yatırımcı"] = ("Önceki yatırımcı", "sum")
        agg["Net akış (TL, tahmini)"] = ("Net akış (TL, tahmini)", "sum")
    g = ft.groupby("Kategori").agg(**agg)
    g["Pazar payı (%)"] = g["Portföy büyüklüğü (TL)"] / g["Portföy büyüklüğü (TL)"].sum()
    g["Yatırımcı payı (%)"] = g["Yatırımcı sayısı"] / g["Yatırımcı sayısı"].sum()
    if "Önceki büyüklük (TL)" in g.columns:
        g["Büyüklük değişimi (%)"] = (g["Portföy büyüklüğü (TL)"] /
                                      g["Önceki büyüklük (TL)"] - 1).where(
            g["Önceki büyüklük (TL)"] > 0)
        g["Yatırımcı değişimi"] = g["Yatırımcı sayısı"] - g["Önceki yatırımcı"]
    return g.sort_values("Portföy büyüklüğü (TL)", ascending=False).replace(
        [np.inf, -np.inf], np.nan)


def selection_summary(ft, codes):
    """Seçilen fonların toplamları: piyasa payı, yatırımcı, tahmini net akış."""
    codes = [c for c in codes if c in ft.index]
    s = ft.loc[codes]
    tot = ft["Portföy büyüklüğü (TL)"].sum()
    size = s["Portföy büyüklüğü (TL)"].sum()
    flow = s["Net akış (TL, tahmini)"].sum() if "Net akış (TL, tahmini)" in s.columns else np.nan
    return {"n": len(s), "size": size, "market_share": size / tot if tot else np.nan,
            "investors": s["Yatırımcı sayısı"].sum(), "flow": flow, "table": s}


# ---------- Profesyonel fon bülteni ----------






















# ---------- Gelişmiş performans ve risk ölçüleri ----------
from statistics import NormalDist
_ND = NormalDist()


def efficiency_table(p, rf):
    """Risk ayarlı getiri göstergeleri (Sharpe, Sortino, Calmar, Omega ...)."""
    rows = {}
    for c in p.columns:
        s = p[c].dropna()
        if len(s) < 30:
            continue
        days = (s.index[-1] - s.index[0]).days
        if days <= 0:
            continue
        r = s.pct_change().dropna()
        ppy = obs_per_year(s)
        rf_d = (1 + rf) ** (1 / ppy) - 1
        total = s.iloc[-1] / s.iloc[0] - 1
        cagr = (1 + total) ** (365.25 / days) - 1
        ex = r - rf_d
        sharpe = ex.mean() / ex.std() * np.sqrt(ppy) if ex.std() > 0 else np.nan
        dsd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
        sortino = ex.mean() / dsd * np.sqrt(ppy) if dsd > 0 else np.nan
        mdd = (s / s.cummax() - 1).min()
        gains, losses = np.maximum(ex, 0).sum(), np.maximum(-ex, 0).sum()
        pos, neg = r[r > 0], r[r < 0]
        rows[c] = {
            "Dönem getirisi": total,
            "Yıllıklandırılmış getiri (CAGR)": cagr,
            "Yıllık volatilite (σ)": r.std() * np.sqrt(ppy),
            "Sharpe oranı": sharpe,
            "Sortino oranı": sortino,
            "Calmar oranı": cagr / abs(mdd) if mdd < 0 else np.nan,
            "Omega oranı": gains / losses if losses > 0 else np.nan,
            "Kazanç/Kayıp oranı": (pos.mean() / abs(neg.mean())
                                   if len(pos) and len(neg) else np.nan),
            "Pozitif gün oranı": (r > 0).mean()}
    return pd.DataFrame(rows).T.astype(float)


def active_table(p, bench, rf):
    """Kıyas fonuna göre beta, alfa, takip hatası, bilgi oranı, yakalama oranları."""
    if bench not in p.columns:
        return pd.DataFrame()
    rows = {}
    for c in p.columns:
        if c == bench:
            continue
        pair = p[[c, bench]].dropna()
        days = (pair.index[-1] - pair.index[0]).days if len(pair) else 0
        if len(pair) < 30 or days <= 0:
            continue
        a = pair.pct_change().dropna()
        rr, rb = a[c], a[bench]
        if rb.var() <= 0:
            continue
        ppy = (len(pair) - 1) / (days / 365.25)
        rf_d = (1 + rf) ** (1 / ppy) - 1
        beta = rr.cov(rb) / rb.var()
        act = rr - rb
        te = act.std() * np.sqrt(ppy)
        up, dn = a[rb > 0], a[rb < 0]
        rows[c] = {
            "Beta (β)": beta,
            "Jensen alfası (α, yıllık)": ((rr - rf_d) - beta * (rb - rf_d)).mean() * ppy,
            "Korelasyon (ρ)": rr.corr(rb),
            "Belirlilik katsayısı (R²)": rr.corr(rb) ** 2,
            "Aktif getiri (yıllık)": act.mean() * ppy,
            "Takip hatası (TE)": te,
            "Bilgi oranı (IR)": act.mean() * ppy / te if te > 0 else np.nan,
            "Treynor oranı": (rr.mean() - rf_d) * ppy / beta if beta != 0 else np.nan,
            "Yukarı yakalama": (up[c].mean() / up[bench].mean()) if len(up) > 5 else np.nan,
            "Aşağı yakalama": (dn[c].mean() / dn[bench].mean()) if len(dn) > 5 else np.nan}
    return pd.DataFrame(rows).T.astype(float)


def drawdown_table(p):
    """Düşüş (drawdown) analizi: derinlik, tarihler, toparlanma ve süre ölçüleri."""
    rows = {}
    fmt = lambda d: d.strftime("%d.%m.%Y")
    for c in p.columns:
        s = p[c].dropna()
        if len(s) < 5:
            continue
        dd = s / s.cummax() - 1
        trough = dd.idxmin()
        peak_date = s.loc[:trough].idxmax()
        after = s.loc[trough:]
        rec = after[after >= s.loc[peak_date]]
        under, idx, n = (dd < 0).values, s.index, len(s)
        longest, i = 0, 0
        while i < n:
            if under[i]:
                j = i
                while j < n and under[j]:
                    j += 1
                start = idx[i - 1] if i > 0 else idx[i]
                end = idx[j] if j < n else idx[-1]
                longest = max(longest, (end - start).days)
                i = j
            else:
                i += 1
        rows[c] = {
            "Maks. düşüş (MDD)": dd.min(),
            "Zirve tarihi": fmt(peak_date),
            "Dip tarihi": fmt(trough),
            "Toparlanma tarihi": fmt(rec.index[0]) if len(rec) else "Devam ediyor",
            "Zirveden dibe (gün)": (trough - peak_date).days,
            "En uzun su altı süresi (gün)": longest,
            "Ortalama düşüş": dd[dd < 0].mean() if (dd < 0).any() else 0.0,
            "Güncel düşüş": dd.iloc[-1],
            "Ulcer endeksi": float(np.sqrt((dd ** 2).mean())),
            "Pain endeksi": float(dd.abs().mean())}
    return pd.DataFrame(rows).T


def volatility_table(p, rf, lam=0.94, window=63):
    """Oynaklık ölçüleri: σ, aşağı sapma, EWMA ve güncel/tarihsel oran."""
    rows = {}
    for c in p.columns:
        s = p[c].dropna()
        if len(s) < 30:
            continue
        r = s.pct_change().dropna()
        ppy = obs_per_year(s)
        rf_d = (1 + rf) ** (1 / ppy) - 1
        vol = r.std() * np.sqrt(ppy)
        ew = np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1]) * np.sqrt(ppy)
        recent = r.tail(window).std() * np.sqrt(ppy) if len(r) >= window else np.nan
        rows[c] = {
            "Günlük standart sapma": r.std(),
            "Yıllık volatilite (σ)": vol,
            "Aşağı sapma (yıllık)": np.sqrt((np.minimum(r - rf_d, 0) ** 2).mean()) * np.sqrt(ppy),
            "EWMA volatilitesi (λ=0,94)": ew,
            f"Son {window} gözlem volatilitesi": recent,
            "Güncel / tarihsel oynaklık": recent / vol if pd.notna(recent) and vol > 0 else np.nan}
    return pd.DataFrame(rows).T.astype(float)


def tail_table(p, alpha, amount):
    """Dağılım şekli, normallik testi ve üç VaR yöntemi (tarihsel, parametrik, Cornish-Fisher)."""
    z = _ND.inv_cdf(1 - alpha)
    rows = {}
    for c in p.columns:
        s = p[c].dropna()
        if len(s) < 30:
            continue
        r = s.pct_change().dropna()
        n, mu = len(r), r.mean()
        sd_p = r.std(ddof=0)
        sd = r.std()
        if sd_p == 0:
            continue
        sk = ((r - mu) ** 3).mean() / sd_p ** 3
        ku = ((r - mu) ** 4).mean() / sd_p ** 4 - 3
        jb = n / 6 * (sk ** 2 + ku ** 2 / 4)
        p_jb = float(np.exp(-jb / 2))            # khi-kare(2) için tam çözüm
        var_h = -np.percentile(r, (1 - alpha) * 100)
        zcf = z + (z ** 2 - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk ** 2 / 36
        rows[c] = {
            "Çarpıklık (S)": sk,
            "Aşırı basıklık (K)": ku,
            "Jarque–Bera p-değeri": p_jb,
            "Normallik (%5)": "Reddedilir" if p_jb < 0.05 else "Reddedilemez",
            "Tarihsel VaR": var_h,
            "Parametrik VaR": -(mu + z * sd),
            "Cornish–Fisher VaR": -(mu + zcf * sd),
            "Tarihsel CVaR (ES)": -r[r <= -var_h].mean(),
            "Parametrik CVaR (ES)": -mu + sd * _ND.pdf(z) / (1 - alpha),
            "Tarihsel VaR (TL)": amount * var_h,
            "En kötü gün": r.min(),
            "En kötü hafta (5g)": s.pct_change(5).dropna().min(),
            "En kötü ay (21g)": s.pct_change(21).dropna().min(),
            "En iyi gün": r.max()}
    return pd.DataFrame(rows).T


def return_histogram(r, th, title=""):
    import altair as alt
    df = pd.DataFrame({"Getiri": r.values})
    ch = alt.Chart(df).mark_bar(color=th["accent"], opacity=0.85).encode(
        x=alt.X("Getiri:Q", bin=alt.Bin(maxbins=40), title="Günlük getiri",
                axis=alt.Axis(format=".1%")),
        y=alt.Y("count():Q", title="Gözlem sayısı"))
    return _style_alt(ch, title, th, height=240)


def show_glossary(entries):
    for name, defn, formula, reading in entries:
        st.markdown(f"**{name}**  \n{defn}  \n*Formül:* `{formula}`  \n*Yorum:* {reading}")


PERF_GLOSSARY = [
    ("Dönem Getirisi (Holding Period Return)",
     "Seçilen dönemde fon birim fiyatındaki toplam değişim.",
     "R = P_T / P_0 − 1",
     "Fon fiyatı bitiş tarihinde başlangıca göre ne kadar değişti. Dönem uzunluğuna bağlıdır, farklı sürelerle doğrudan kıyaslanmaz."),
    ("Yıllıklandırılmış Bileşik Getiri (CAGR)",
     "Dönem getirisinin takvim gününe göre yıllık bileşik oranı.",
     "CAGR = (1 + R)^(365,25 / gün) − 1",
     "Farklı sürelerdeki fonları yıllık bazda kıyaslamayı sağlar. 90 günden kısa dönemlerde yıllıklandırma yanıltıcı olabilir."),
    ("Sharpe Oranı",
     "Toplam riskin (volatilite) her birimi başına risksiz oranın üzerinde elde edilen getiri.",
     "SR = ort(r − rf) / σ(r − rf) × √(yıllık gözlem)",
     "Yüksek değer daha iyi risk ayarlı getiridir. Negatifse fon, risksiz oranın bile altında getiri üretmiştir. Sonuç, girilen risksiz orana duyarlıdır."),
    ("Sortino Oranı",
     "Sharpe'ın yalnızca aşağı yönlü oynaklığı riske sayan türü. Eşik olarak risksiz oran (MAR) kullanılır.",
     "Sortino = ort(r − MAR) / AS × √(yıllık gözlem), AS = √ort(min(r − MAR, 0)²)",
     "Yükseliş oynaklığını cezalandırmaz. Sharpe'tan belirgin yüksekse oynaklığın büyük kısmı yukarı yönlüdür."),
    ("Calmar Oranı",
     "Yıllıklandırılmış getirinin maksimum düşüşe oranı.",
     "Calmar = CAGR / |MDD|",
     "Katlanılan en büyük kayba karşılık elde edilen yıllık getiri. Yüksek değer, kayıpların getiriyle hızlı telafi edildiğini gösterir."),
    ("Omega Oranı",
     "Eşiğin (risksiz getiri) üzerindeki toplam kazancın, altındaki toplam kayba oranı.",
     "Ω = Σ max(r − τ, 0) / Σ max(τ − r, 0)",
     "1'in üzeri, eşiği aşan kazançların eşik altı kayıplardan büyük olduğunu gösterir. Dağılımın tüm momentlerini kullanır."),
    ("Kazanç/Kayıp Oranı",
     "Ortalama pozitif günlük getirinin, ortalama negatif günlük getirinin mutlak değerine oranı.",
     "G/K = ort(r | r > 0) / |ort(r | r < 0)|",
     "1'in üzeri, kazanç günlerinin kayıp günlerinden büyük olduğunu gösterir; pozitif gün oranıyla birlikte okunmalıdır."),
    ("Pozitif Gün Oranı (Hit Rate)",
     "Getirinin sıfırın üzerinde olduğu gözlemlerin payı.",
     "HR = #(r > 0) / n",
     "Getirinin tutarlılığını gösterir; büyüklüğü hakkında bilgi vermez."),
    ("Beta (β)",
     "Fon getirisinin kıyas getirisindeki değişime duyarlılığı.",
     "β = Kov(r_f, r_k) / Var(r_k)",
     "β = 1 kıyasla aynı yönde ve oranda hareket; 1'den büyük kıyastan daha oynak, 1'den küçük daha savunmacıdır."),
    ("Jensen Alfası (α)",
     "Kıyasa duyarlılık (β) arındırıldıktan sonra yöneticinin yarattığı artık getiri (yıllık).",
     "α = ort[(r_f − rf) − β(r_k − rf)] × yıllık gözlem",
     "Pozitif α, fonun piyasa riskinin açıkladığından fazla getiri ürettiğini gösterir. İstatistiksel anlamlılığı ayrıca test edilmemiştir."),
    ("Korelasyon (ρ) ve Belirlilik Katsayısı (R²)",
     "Fon ile kıyas getirileri arasındaki doğrusal ilişki ve bu ilişkinin açıkladığı varyans oranı.",
     "ρ = Kor(r_f, r_k), R² = ρ²",
     "R² yüksekse fonun getirisi büyük ölçüde kıyasın hareketiyle açıklanır; düşükse fon kıyastan bağımsız bir getiri profili taşır."),
    ("Aktif Getiri",
     "Fon getirisinin kıyas getirisinden ortalama farkı (yıllık).",
     "AG = ort(r_f − r_k) × yıllık gözlem",
     "Pozitif değer, fonun kıyası ortalamada geçtiğini gösterir."),
    ("Takip Hatası (Tracking Error, TE)",
     "Aktif getirinin standart sapması (yıllık).",
     "TE = σ(r_f − r_k) × √(yıllık gözlem)",
     "Fonun kıyastan ne kadar saptığını gösterir. Düşük TE kıyasa yakın, yüksek TE aktif yönetimi ifade eder."),
    ("Bilgi Oranı (Information Ratio, IR)",
     "Aktif getirinin takip hatası başına büyüklüğü.",
     "IR = AG / TE",
     "Yöneticinin aldığı aktif riske karşılık ürettiği getiri. 0,5 üzeri genellikle iyi kabul edilir."),
    ("Treynor Oranı",
     "Sistematik risk (β) başına risksiz orana göre fazla getiri.",
     "Treynor = ort(r_f − rf) × yıllık gözlem / β",
     "Çeşitlendirilmiş portföyler için Sharpe'ın alternatifidir. β sıfıra yakınsa anlamsızlaşır."),
    ("Yukarı / Aşağı Yakalama Oranı (Up / Down Capture)",
     "Kıyasın yükseldiği (düştüğü) günlerde fonun ortalama getirisinin kıyasınkine oranı.",
     "YY = ort(r_f | r_k > 0) / ort(r_k | r_k > 0); AY = ort(r_f | r_k < 0) / ort(r_k | r_k < 0)",
     "İdeal profil YY > 1 ve AY < 1'dir: yükselişi yakalar, düşüşten daha az etkilenir."),
    ("Düşüş (Drawdown) ve Maksimum Düşüş (MDD)",
     "Fiyatın o ana kadarki zirvesinden yüzde sapması; MDD bunun en derin değeridir.",
     "D_t = P_t / maks(P_s, s ≤ t) − 1; MDD = min(D_t)",
     "Yatırımcının zirveden yaşayabileceği en kötü tarihsel kaybı gösterir."),
    ("Su Altı Süresi (Time Under Water)",
     "Fiyatın önceki zirvenin altında kaldığı en uzun süre (takvim günü).",
     "zirve tarihi → toparlanma tarihi",
     "Kaybın ne kadar sürede telafi edildiğini gösterir; 'Devam ediyor' fonun hâlâ zirvenin altında olduğunu belirtir."),
    ("Ulcer ve Pain Endeksi",
     "Düşüşlerin hem derinliğini hem süresini yansıtan ortalama ölçüler.",
     "Ulcer = √ort(D_t²); Pain = ort(|D_t|)",
     "Yüksek değer, fonun uzun süre ve derin biçimde zirvenin altında kaldığını gösterir."),
]

RISK_GLOSSARY = [
    ("Yıllık Volatilite (σ)",
     "Günlük getirilerin standart sapmasının yıllığa ölçeklenmiş hali.",
     "σ = std(r) × √(yıllık gözlem)",
     "Getirinin ortalama etrafındaki dağılımını gösterir; yüksek σ daha oynak fon demektir."),
    ("Aşağı Sapma (Downside Deviation)",
     "Yalnızca eşiğin (risksiz oran) altındaki getirilerin dalgalanması.",
     "AS = √ort(min(r − MAR, 0)²) × √(yıllık gözlem)",
     "Yükseliş oynaklığını riskten saymaz; yatırımcının gerçekten kaçınmak istediği riski ölçer."),
    ("EWMA Volatilitesi (RiskMetrics)",
     "Yakın dönem gözlemlere daha fazla ağırlık veren üstel ağırlıklı oynaklık tahmini.",
     "σ²_t = λ σ²_(t−1) + (1 − λ) r²_(t−1), λ = 0,94",
     "Güncel risk rejimini hızlı yakalar. Tarihsel σ'dan yüksekse risk son dönemde artmıştır."),
    ("Güncel / Tarihsel Oynaklık",
     "Son 63 gözlemin (yaklaşık 3 ay) volatilitesinin tüm dönem volatilitesine oranı.",
     "σ_63 / σ_tüm",
     "1'in üzeri riskin arttığını, altı sakinleştiğini gösterir."),
    ("Çarpıklık (Skewness, S)",
     "Getiri dağılımının simetrisizliği.",
     "S = E[(r − μ)³] / σ³",
     "Negatif S sol kuyruğun ağır olduğunu, yani büyük kayıp olasılığının büyük kazançtan yüksek olduğunu gösterir."),
    ("Aşırı Basıklık (Excess Kurtosis, K)",
     "Dağılımın kuyruk kalınlığı; normal dağılım için 0'dır.",
     "K = E[(r − μ)⁴] / σ⁴ − 3",
     "K > 0 kalın kuyruk: uç hareketler normal dağılımın öngördüğünden sık görülür."),
    ("Jarque–Bera Normallik Testi",
     "Çarpıklık ve basıklığa göre getirilerin normal dağılıma uyup uymadığını sınar.",
     "JB = n/6 · (S² + K²/4); p = exp(−JB/2)",
     "p < 0,05 ise normallik reddedilir; bu durumda parametrik (Gauss) VaR kuyruk riskini olduğundan düşük gösterebilir."),
    ("Tarihsel VaR (Riske Maruz Değer)",
     "Gözlenen getirilerin (1 − α) yüzdelik dilimi; dağılım varsayımı yapmaz.",
     "VaR_α = −Q_(1−α)(r)",
     "Bir günde, %α olasılıkla bu oranı aşan kayıp yaşanmamıştır (tarihsel olarak). Eşik aşıldığında kaybın ne kadar büyüyeceğini söylemez."),
    ("Parametrik (Varyans-Kovaryans) VaR",
     "Getirilerin normal dağıldığı varsayımıyla ortalama ve standart sapmadan hesaplanan VaR.",
     "VaR = −(μ + z_(1−α) · σ)",
     "Hızlı ve istikrarlıdır, ancak kalın kuyruklu fonlarda riski eksik tahmin edebilir."),
    ("Cornish–Fisher (Modifiye) VaR",
     "Normal VaR'ın çarpıklık ve basıklığa göre düzeltilmiş hali.",
     "z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36; VaR = −(μ + z_cf · σ)",
     "Parametrik VaR'dan belirgin büyükse dağılımın kuyruk riski normalin öngördüğünden fazladır."),
    ("CVaR / Beklenen Kayıp (Expected Shortfall, ES)",
     "VaR eşiği aşıldığında beklenen ortalama kayıp.",
     "ES = −E[r | r ≤ −VaR]",
     "VaR'ın söylemediği 'kötü gün ne kadar kötü olur' sorusunu yanıtlar; kuyruk riskinin daha tutarlı bir ölçüsüdür."),
    ("En Kötü Gün / Hafta / Ay",
     "Gözlem döneminde görülen en düşük 1, 5 ve 21 gözlemlik getiri.",
     "min(r_1g), min(r_5g), min(r_21g)",
     "Tarihsel stres göstergesidir; geleceğin alt sınırı değildir."),
    ("Korelasyon Matrisi ve Çeşitlendirme",
     "Fonların günlük getirileri arasındaki doğrusal ilişki (−1 ile +1).",
     "ρ_ij = Kor(r_i, r_j)",
     "Düşük veya negatif korelasyon çeşitlendirme faydası sağlar; yüksek korelasyonlu fonlar sepette benzer riski tekrarlar."),
]


# ---------- Arayüz ----------
with st.sidebar:
    st.header("Ayarlar")
    title = st.text_input("Panel başlığı", "Fon Analiz Paneli")
    tema = st.selectbox("Görünüm", list(THEMES))
    st.session_state.setdefault("codes_in", "AFA,TI2,IPB,YAS")
    codes = [c.strip().upper() for c in
             st.text_input("Fon kodları (virgülle)", key="codes_in").split(",")
             if c.strip()]
    start = st.date_input("Başlangıç", pd.Timestamp.today() - pd.DateOffset(years=1))
    end = st.date_input("Bitiş", pd.Timestamp.today())
    live = st.toggle("Canlı TEFAS verisi", value=True,
                     help="Kapalıyken sentetik demo verisi kullanılır.")
    rf = st.number_input(
        "Yıllık risksiz oran (%)", 0.0, 100.0, 40.0, 0.25,
        help="Sharpe ve fazla getiri hesabında kullanılır. Güncel politika faizi veya "
             "mevduat oranını gir; varsayılan yer tutucudur.") / 100
    alpha = st.selectbox(
        "VaR güven düzeyi", [0.90, 0.95, 0.99], index=1,
        format_func=lambda a: f"%{int(a * 100)}",
        help="1 günlük VaR: tarihsel simülasyonla, seçilen olasılıkta bir günde "
             "beklenen en kötü kayıp eşiği. Yüksek düzey daha uç kayıpları gösterir.")
    st.caption("VaR ve CVaR: 1 günlük, tarihsel simülasyon.")

th = THEMES[tema]
st.markdown(make_css(th), unsafe_allow_html=True)
pill = ("<span class='pill live'>● CANLI TEFAS VERİSİ</span>" if live
        else "<span class='pill demo'>● DEMO VERİ</span>")
st.markdown(
    f"<div class='hdr'><div><div class='brand'>{title}</div>"
    "<div class='sub'>Fon performans, risk ve akış analizi</div></div>"
    f"<div>{pill}<span class='date'>{pd.Timestamp.today():%d.%m.%Y}</span></div></div>",
    unsafe_allow_html=True)
if not codes:
    st.stop()

try:
    raw = (load_tefas(tuple(codes), str(start), str(end)) if live
           else load_synthetic(tuple(codes), str(start), str(end)))
    w = to_wide(raw)
except Exception as e:
    st.error(f"Veri alınamadı: {e}")
    st.stop()

p = w["price"]
if p.empty:
    st.warning("Seçilen tarih aralığında veri yok. Fon kodlarını kontrol et.")
    st.stop()

bench = st.sidebar.selectbox("Kıyas fonu (tracking error)", list(p.columns))
st.caption(f"Kaynak: {'TEFAS (canlı)' if live else 'sentetik demo verisi'} | "
           f"Son veri tarihi: {p.index[-1]:%d.%m.%Y}")
pf = p.ffill()  # yalnızca grafik çizimi için

titles_all = {}
if "title" in raw.columns:
    titles_all = (raw.dropna(subset=["title"]).drop_duplicates("code")
                  .set_index("code")["title"].to_dict())
NAMES = ['Home', 'Portföy sepeti', 'Yatırımcı karşılaştırma', 'Performans', 'Risk', 'Varlık dağılımı', 'Fon büyüklüğü ve akış', 'ETF', 'ETF Analiz', 'ETF Detay']
T = dict(zip(NAMES, st.tabs(NAMES)))

with T["Home"]:
    hm = st.tabs(["Genel bakış", "Yatırımcı profili", "Öneri ve örnek portföy"])
    with hm[0]:
        st.markdown(ticker_html(p), unsafe_allow_html=True)
        ft_h = None
        if live:
            try:
                cur_h, ref_h, _, _ = load_fund_info(str(pd.Timestamp.today().date()))
                ft_h = fund_table(cur_h, ref_h)
            except Exception as e:
                st.caption(f"TEFAS fon bilgisi alınamadı: {e}")
        k1, k2, k3 = st.columns(3)
        if ft_h is not None:
            k1.metric("Piyasadaki fon", f"{len(ft_h):,}")
            k2.metric("Toplam fon büyüklüğü",
                      f"{ft_h['Portföy büyüklüğü (TL)'].sum() / 1e9:,.2f} mlr TL")
            k3.metric("Toplam yatırımcı", f"{ft_h['Yatırımcı sayısı'].sum():,.0f}")
        else:
            k1.metric("İzlenen fon", f"{len(p.columns)}")
            k2.metric("Toplam fon büyüklüğü", "-")
            k3.metric("Toplam yatırımcı", "-")

        h1, h2 = st.columns(2)
        with h1:
            st.markdown("##### Fiyat performansı (başlangıç = 100)")
            base_h = p.apply(lambda s: s.dropna().iloc[0])
            st.altair_chart(line_chart(pf / base_h * 100, th, "", ",.0f"),
                            use_container_width=True, theme=None)
        with h2:
            st.markdown("##### 1 aylık getiri")
            pr_h = period_returns(p)
            bm = pr_h["1 Ay"].dropna().astype(float).rename("Getiri").rename_axis("Fon").reset_index()
            if len(bm):
                st.altair_chart(bar_pos_neg(bm, "Fon", "Getiri", "", th),
                                use_container_width=True, theme=None)

        h3, h4 = st.columns(2)
        with h3:
            st.markdown("##### Risk / getiri haritası")
            rt_h = risk_table(p, bench, rf, alpha)
            if not rt_h.empty:
                sc = (rt_h[["Yıllık volatilite", "Yıllık getiri"]].astype(float)
                      .rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
                      .rename_axis("Fon").reset_index())
                st.altair_chart(scatter_chart(sc, th, ""), use_container_width=True, theme=None)
        with h4:
            st.markdown("##### Kategoriye göre pazar payı")
            if ft_h is not None:
                ct_h = category_table(ft_h)
                dn = ct_h.reset_index()[["Kategori", "Portföy büyüklüğü (TL)"]]
                dn.columns = ["Kategori", "Deger"]
                st.altair_chart(donut_chart(dn, "Kategori", "Deger", th),
                                use_container_width=True, theme=None)
            else:
                st.info("Kategori dağılımı için canlı TEFAS verisi gerekir.")
    with hm[1]:
        render_profil(globals())
    with hm[2]:
        render_oneri(globals())

with T["Performans"]:
    st.subheader("Performans Analizi")
    st.caption(f"Risksiz oran / MAR: %{rf * 100:.2f} · Kıyas: {bench} · "
               f"Gözlem aralığı: {p.index[0]:%d.%m.%Y} – {p.index[-1]:%d.%m.%Y}")
    pt = st.tabs(["Getiri Analizi", "Risk Ayarlı Getiri", "Kıyasa Göre Aktif Performans",
                  "Düşüş (Drawdown) Analizi", "Ölçüt Sözlüğü"])
    with pt[0]:
        st.markdown("##### Birim Fiyat Gelişimi (Başlangıç = 100)")
        base_p = p.apply(lambda s: s.dropna().iloc[0])
        st.altair_chart(line_chart(pf / base_p * 100, th, "", ",.0f"),
                        use_container_width=True, theme=None)
        st.markdown("##### Dönemsel Getiri Tablosu ve Sıralama")
        prt = period_returns(p)
        st.dataframe(prt.style.format({c: "{:+.2%}" for c in prt.columns
                                       if c != "Sıra (3 Ay)"}, na_rep="-"),
                     use_container_width=True)
        mr = monthly_returns(p)
        if len(mr):
            st.markdown("##### Aylık Getiri Isı Haritası")
            st.altair_chart(monthly_heatmap(mr, th), use_container_width=True, theme=None)
        st.markdown("##### Kayan 21 Gözlemlik (≈ 1 Aylık) Getiri")
        st.altair_chart(line_chart(pf.pct_change(21).dropna(how="all"), th, "", ".0%"),
                        use_container_width=True, theme=None)
        st.markdown("##### Getiri Dağılımı (Histogram)")
        hf = st.selectbox("Fon", list(p.columns), key="pf_hist")
        rh = p[hf].dropna().pct_change().dropna()
        if len(rh) >= 10:
            st.altair_chart(return_histogram(rh, th), use_container_width=True, theme=None)
    with pt[1]:
        st.markdown("##### Risk Ayarlı Getiri Göstergeleri")
        et = efficiency_table(p, rf)
        if et.empty:
            st.info("Bu tablo için fon başına en az 30 gözlem gerekir.")
        else:
            efmt = {c: "{:+.2%}" for c in ["Dönem getirisi", "Yıllıklandırılmış getiri (CAGR)"]}
            efmt.update({"Yıllık volatilite (σ)": "{:.2%}", "Pozitif gün oranı": "{:.1%}"})
            efmt.update({c: "{:.2f}" for c in ["Sharpe oranı", "Sortino oranı", "Calmar oranı",
                                               "Omega oranı", "Kazanç/Kayıp oranı"]})
            st.dataframe(et.style.format(efmt, na_rep="-"), use_container_width=True)
            st.download_button("Tabloyu indir (CSV)", et.to_csv().encode("utf-8-sig"),
                               "risk_ayarli_getiri.csv", "text/csv")
            st.caption("Sharpe, Sortino ve Omega'da eşik olarak yan paneldeki risksiz oran kullanılır.")
    with pt[2]:
        rel = relative_table(p, bench)
        if rel is not None and len(rel):
            st.markdown(f"##### Dönemsel Getiri Farkı (Fon − {bench}), Yüzde Puan")
            st.dataframe(rel.style.format("{:+.2%}", na_rep="-"), use_container_width=True)
        st.markdown(f"##### Aktif Performans Göstergeleri (Kıyas: {bench})")
        at = active_table(p, bench, rf)
        if at.empty:
            st.info("Kıyas fonundan farklı en az bir fon ve yeterli ortak gözlem gerekir.")
        else:
            afmt = {"Beta (β)": "{:.2f}", "Jensen alfası (α, yıllık)": "{:+.2%}",
                    "Korelasyon (ρ)": "{:.2f}", "Belirlilik katsayısı (R²)": "{:.2f}",
                    "Aktif getiri (yıllık)": "{:+.2%}", "Takip hatası (TE)": "{:.2%}",
                    "Bilgi oranı (IR)": "{:.2f}", "Treynor oranı": "{:.2f}",
                    "Yukarı yakalama": "{:.2f}", "Aşağı yakalama": "{:.2f}"}
            st.dataframe(at.style.format(afmt, na_rep="-"), use_container_width=True)
    with pt[3]:
        st.markdown("##### Su Altı (Underwater) Grafiği: Zirveden Düşüş")
        st.altair_chart(line_chart(pf / pf.cummax() - 1, th, "", ".0%"),
                        use_container_width=True, theme=None)
        st.markdown("##### Düşüş Analizi Tablosu")
        dtb = drawdown_table(p)
        if not dtb.empty:
            dfmt = {"Maks. düşüş (MDD)": "{:.2%}", "Ortalama düşüş": "{:.2%}",
                    "Güncel düşüş": "{:.2%}", "Ulcer endeksi": "{:.2%}",
                    "Pain endeksi": "{:.2%}", "Zirveden dibe (gün)": "{:,.0f}",
                    "En uzun su altı süresi (gün)": "{:,.0f}"}
            st.dataframe(dtb.style.format(dfmt, na_rep="-"), use_container_width=True)
    with pt[4]:
        st.markdown("##### Performans ve Risk Ayarlı Getiri Ölçütleri")
        show_glossary(PERF_GLOSSARY)

with T["Risk"]:
    st.subheader("Risk Analizi")
    st.caption(f"VaR güven düzeyi: %{int(alpha * 100)} · Ufuk: 1 gözlem (günlük) · "
               f"Risksiz oran / MAR: %{rf * 100:.2f}")
    rk = st.tabs(["Oynaklık Analizi", "Kuyruk Riski ve VaR", "Korelasyon ve Çeşitlendirme",
                  "Ölçüt Sözlüğü"])
    with rk[0]:
        st.markdown("##### Oynaklık (Volatilite) Ölçüleri")
        vt = volatility_table(p, rf)
        if vt.empty:
            st.info("Bu tablo için fon başına en az 30 gözlem gerekir.")
        else:
            st.dataframe(vt.style.format(
                {c: ("{:.2f}" if "tarihsel" in c else "{:.2%}") for c in vt.columns},
                na_rep="-"), use_container_width=True)
        st.markdown("##### Kayan 30 Gözlemlik Volatilite (Yıllıklandırılmış)")
        Rv = returns_df(p)
        rvv = Rv.rolling(30, min_periods=20).std() * np.sqrt(
            pd.Series({c: obs_per_year(p[c]) for c in p}))
        st.altair_chart(line_chart(rvv, th, "", ".0%"), use_container_width=True, theme=None)
    with rk[1]:
        amt = st.number_input("Risk tutarı için yatırım tutarı (TL)", 1000.0, 1e12, 100000.0,
                              1000.0, key="rk_amt")
        st.markdown(f"##### Dağılım Şekli ve Riske Maruz Değer (VaR / CVaR), %{int(alpha * 100)} Güven")
        tt = tail_table(p, alpha, amt)
        if tt.empty:
            st.info("Bu tablo için fon başına en az 30 gözlem gerekir.")
        else:
            tfm = {c: "{:.2%}" for c in tt.columns if c not in
                   ("Çarpıklık (S)", "Aşırı basıklık (K)", "Jarque–Bera p-değeri",
                    "Normallik (%5)", "Tarihsel VaR (TL)")}
            tfm.update({"Çarpıklık (S)": "{:.2f}", "Aşırı basıklık (K)": "{:.2f}",
                        "Jarque–Bera p-değeri": "{:.4f}", "Tarihsel VaR (TL)": "{:,.0f}"})
            st.dataframe(tt.style.format(tfm, na_rep="-"), use_container_width=True)
            st.download_button("Tabloyu indir (CSV)", tt.to_csv().encode("utf-8-sig"),
                               "kuyruk_riski_var.csv", "text/csv")
            st.caption("Parametrik ve Cornish–Fisher VaR tarihsel VaR'dan belirgin farklıysa "
                       "getiri dağılımı normal varsayımından sapıyor demektir.")
    with rk[2]:
        st.markdown("##### Korelasyon Matrisi (Ortak Tarihler, Günlük Getiri)")
        st.altair_chart(corr_heatmap(returns_df(p).corr(), th), use_container_width=True,
                        theme=None)
        st.markdown("##### Risk / Getiri Haritası")
        rt_s = risk_table(p, bench, rf, alpha)
        if not rt_s.empty:
            sc2 = (rt_s[["Yıllık volatilite", "Yıllık getiri"]].astype(float)
                   .rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
                   .rename_axis("Fon").reset_index())
            st.altair_chart(scatter_chart(sc2, th, ""), use_container_width=True, theme=None)
    with rk[3]:
        st.markdown("##### Risk Ölçütleri")
        show_glossary(RISK_GLOSSARY)

with T["Portföy sepeti"]:
    st.subheader("Portföy sepeti")
    st.caption("Her fona yatıracağın tutarı gir; dağılım, risk ve geçmiş performans "
               "birlikte gösterilir.")
    if "bk_pick" not in st.session_state:
        st.session_state["bk_pick"] = list(p.columns)[:3]
    st.session_state["bk_pick"] = [c for c in st.session_state["bk_pick"] if c in p.columns]
    bk_pick = st.multiselect("Sepetteki fonlar", list(p.columns), key="bk_pick")
    if bk_pick:
        dly = returns_df(p)
        inv_map = {}
        if live:
            try:
                cur_s, _, _, _ = load_fund_info(str(pd.Timestamp.today().date()))
                inv_map = cur_s.drop_duplicates("fund_code").set_index("fund_code")["investor_count"].to_dict()
            except Exception:
                inv_map = {}
        rows_d = {}
        for c in bk_pick:
            if c in p.columns and p[c].dropna().size > 1:
                rr = dly[c].dropna()
                rows_d[c] = {"Son fiyat": float(p[c].dropna().iloc[-1]),
                             "Günlük getiri": float(rr.iloc[-1]) if len(rr) else np.nan,
                             "Yatırımcı sayısı": float(inv_map.get(c, np.nan))}
        if rows_d:
            st.markdown("##### Seçilen fonların son gün durumu")
            dd_tbl = pd.DataFrame(rows_d).T.astype(float)
            st.dataframe(dd_tbl.style.format({"Son fiyat": "{:,.4f}", "Günlük getiri": "{:+.2%}",
                                              "Yatırımcı sayısı": "{:,.0f}"}),
                         use_container_width=True)
    _uni = etf_secenekleri()
    st.session_state["bk_etf"] = [c for c in st.session_state.get("bk_etf", []) if c in _uni]
    bk_etf = st.multiselect("Sepete ETF ekle (ayrı seçim)", list(_uni),
                            format_func=lambda t: f"{t} · {_uni[t][0]}", key="bk_etf",
                            help="ETF'ler USD/TRY kuruyla TL'ye çevrilir; gider oranı otomatik gelir.")
    bk_per = st.radio("Geçmiş dönem", list(PERIODS), index=3, horizontal=True, key="bk_per")
    amounts = {}
    if bk_pick:
        cols_b = st.columns(min(len(bk_pick), 4))
        for i, c in enumerate(bk_pick):
            st.session_state.setdefault(f"bk_amt_{c}", 50000.0)
            amounts[c] = cols_b[i % len(cols_b)].number_input(
                f"{c} tutarı (TL)", 0.0, 1e12, step=1000.0, key=f"bk_amt_{c}")
    pb = p
    if bk_etf:
        try:
            _ep = etf_tl_fiyat(tuple(bk_etf), start, end)
            bk_etf = [t for t in bk_etf if t in _ep.columns]
            pb = pd.concat([p, _ep[bk_etf]], axis=1).sort_index()
        except Exception as e:
            st.warning(f"ETF verisi alınamadı, sepetten çıkarıldı: {e}")
            bk_etf = []
        if bk_etf:
            cols_e = st.columns(min(len(bk_etf), 4))
            for i, c in enumerate(bk_etf):
                st.session_state.setdefault(f"bk_eamt_{c}", 50000.0)
                amounts[c] = cols_e[i % len(cols_e)].number_input(
                    f"{c} tutarı (TL)", 0.0, 1e12, step=1000.0, key=f"bk_eamt_{c}")
    er_in, e_use, bk_ebuy, bk_esell = {}, True, 0.0, 0.0
    rates_b = {}
    with st.expander("Stopaj ve masraf ayarları"):
        st.caption("Stopaj oranı fon adından tahmin edilir; izahnameden teyit edip "
                   "değiştirebilirsin. Fon yönetim ücreti fiyata dahildir.")
        q1, q2 = st.columns(2)
        bk_buy = q1.number_input("Alım komisyonu (%)", 0.0, 10.0, 0.0, 0.05, key="bk_buy") / 100
        bk_sell = q2.number_input("Satım komisyonu (%)", 0.0, 10.0, 0.0, 0.05, key="bk_sell") / 100
        for c in bk_pick:
            rate_b, kind_b = auto_stopaj(titles_all.get(c, ""))
            rates_b[c] = st.number_input(f"{c} stopaj oranı (%): {kind_b}", 0.0, 50.0,
                                         rate_b * 100, 0.5, key=f"bk_stp_{c}") / 100
        if bk_etf:
            st.markdown("**ETF masrafları**")
            e_use = st.checkbox(
                "ETF yıllık gider oranını getiriden otomatik düş", value=True, key="bk_eer_use",
                help="Yahoo fiyatları gider oranını zaten yansıtır; bu kutu çift sayıma yol "
                     "açabilir. Gider oranını ayrıca düşmek istemezseniz kapatın.")
            f1, f2 = st.columns(2)
            bk_ebuy = f1.number_input("ETF alım komisyonu (%)", 0.0, 10.0, 0.0, 0.05,
                                      key="bk_ebuy") / 100
            bk_esell = f2.number_input("ETF satım komisyonu (%)", 0.0, 10.0, 0.0, 0.05,
                                       key="bk_esell") / 100
            st.caption("Komisyon ve vergi oranlarını aracı kurumunuza ve vergi durumunuza "
                       "göre girin; varsayılan 0'dır.")
            for c in bk_etf:
                auto = etf_gider_orani(c)
                er_in[c] = st.number_input(
                    f"{c} yıllık gider oranı (%): "
                    + ("Yahoo'dan otomatik" if auto is not None else "veri yok, elle girin"),
                    0.0, 5.0, (auto or 0.0) * 100, 0.01, key=f"bk_er_{c}") / 100
                rates_b[c] = st.number_input(f"{c} stopaj/vergi oranı (%)", 0.0, 50.0, 0.0, 0.5,
                                             key=f"bk_estp_{c}") / 100
    if not (bk_pick or bk_etf) or sum(amounts.values()) <= 0:
        st.info("En az bir fon seç ve tutar gir.")
    else:
        _bf = {c: bk_buy for c in bk_pick}
        _bf.update({c: bk_ebuy for c in bk_etf})
        _sf = {c: bk_sell for c in bk_pick}
        _sf.update({c: bk_esell for c in bk_etf})
        res = basket_analysis(pb, amounts, PERIODS[bk_per], rates_b, _bf, _sf, alpha,
                              er_in if e_use else None)
        if res is None:
            st.warning("Seçilen fonların ortak tarihlerinde yeterli veri yok.")
        else:
            tb = res["table"]
            tot_b = tb["Tutar (TL)"].sum()
            net_b = tb["Net değer (TL)"].sum()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam tutar", tl(tot_b))
            m2.metric("Brüt getiri", pc(res["ret"]))
            m3.metric("Net elde edilen", tl(net_b), f"{(net_b / tot_b - 1) * 100:+.2f}%")
            m4.metric("Çeşitlendirme oranı",
                      "-" if pd.isna(res["div"]) else f"{res['div']:.2f}")
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Yıllık volatilite", pc(res["vol"]))
            m6.metric("Maks. düşüş", pc(res["dd"]))
            m7.metric(f"VaR %{int(alpha * 100)} (1g)", tl(res["var"] * tot_b))
            m8.metric(f"CVaR %{int(alpha * 100)} (1g)", tl(res["cvar"] * tot_b))

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("##### Tutar dağılımı")
                dd1 = tb["Tutar (TL)"].rename("Deger").rename_axis("Fon").reset_index()
                st.altair_chart(donut_chart(dd1, "Fon", "Deger", th),
                                use_container_width=True, theme=None)
            with d2:
                st.markdown("##### Risk katkısı dağılımı")
                dd2 = (tb["Risk katkısı"].clip(lower=0).rename("Deger")
                       .rename_axis("Fon").reset_index())
                if dd2["Deger"].sum() > 0:
                    st.altair_chart(donut_chart(dd2, "Fon", "Deger", th),
                                    use_container_width=True, theme=None)
                else:
                    st.info("Risk katkısı hesaplanamadı.")

            e1, e2 = st.columns(2)
            with e1:
                st.markdown("##### Sepet değeri (TL), fon bazında")
                st.altair_chart(stacked_area(res["value"], th, "", ".2s"),
                                use_container_width=True, theme=None)
            with e2:
                st.markdown("##### Fon bazında net kâr/zarar (TL)")
                pf_b = (tb["Net kâr/zarar (TL)"].rename("Sonuc")
                        .rename_axis("Fon").reset_index())
                st.altair_chart(bar_pos_neg(pf_b, "Fon", "Sonuc", "", th, ".2s"),
                                use_container_width=True, theme=None)

            st.markdown("##### Yazılı değerlendirme")
            lab_b = ("seçili veri aralığı" if PERIODS[bk_per] is None
                     else f"son {bk_per.lower()}")
            for para in basket_text(res, lab_b):
                st.markdown(para)
            fmt_b = {"Tutar (TL)": "{:,.0f}", "Ağırlık": "{:.1%}", "Dönem getirisi": "{:.2%}",
                     "Risk katkısı": "{:.1%}", "Brüt değer (TL)": "{:,.0f}",
                     "Stopaj (TL)": "{:,.0f}", "Masraf (TL)": "{:,.0f}",
                     "Net değer (TL)": "{:,.0f}", "Net kâr/zarar (TL)": "{:,.0f}",
                     "Gider oranı (TL)": "{:,.0f}"}
            st.dataframe(tb.style.format(fmt_b), use_container_width=True)
            st.download_button("Tabloyu indir (CSV)", tb.to_csv().encode("utf-8-sig"),
                               "sepet.csv", "text/csv")

with T["Varlık dağılımı"]:
    st.subheader("Varlık dağılımı")
    src_va = st.radio("Veri kaynağı", ["TEFAS (otomatik)", "Manuel giriş"], index=0,
                      horizontal=True, key="va_src")
    va_funds = st.multiselect("Fonlar", list(p.columns), default=list(p.columns)[:3],
                              key="va_funds")
    tbl = None
    if src_va == "TEFAS (otomatik)":
        if not live:
            st.info("Demo modunda TEFAS dağılımı yok. Canlı TEFAS verisini aç ya da 'Manuel giriş' seç.")
        elif va_funds:
            try:
                raw_b = load_breakdown(str(pd.Timestamp.today().date()))
                tbl = breakdown_table(raw_b, va_funds)
                st.caption(f"Kaynak: TEFAS portföy dağılımı | Tarih: "
                           f"{raw_b['date'].astype(str).max()}" if "date" in raw_b.columns
                           else "Kaynak: TEFAS portföy dağılımı")
                if tbl.empty:
                    st.warning("Seçilen fonlar için dağılım verisi bulunamadı.")
                    tbl = None
            except Exception as e:
                st.error(f"Dağılım verisi alınamadı: {e}. requirements.txt dosyasına "
                         "'pytefas' eklenmiş olmalı.")
    else:
        if va_funds:
            base = st.session_state.get("va_base", {})
            init = pd.DataFrame(
                [[base.get(f, {}).get(c, 0.0) for c in ASSET_CLASSES] for f in va_funds],
                index=va_funds, columns=ASSET_CLASSES)
            edited = st.data_editor(
                init, use_container_width=True, key="va_editor",
                column_config={c: st.column_config.NumberColumn(c, min_value=0.0,
                                                                max_value=100.0, format="%.1f")
                               for c in ASSET_CLASSES})
            merged = dict(base)
            merged.update({f: row.to_dict() for f, row in edited.iterrows()})
            st.session_state["va_base"] = merged
            tbl = edited.loc[:, (edited.fillna(0) != 0).any()]
            if tbl.empty:
                tbl = None

    if tbl is not None and len(tbl):
        v1, v2 = st.columns([2, 1])
        with v1:
            st.markdown("##### Fonlara göre dağılım (%)")
            st.altair_chart(alloc_bar(tbl, th), use_container_width=True, theme=None)
        with v2:
            st.markdown("##### Tek fon")
            sel_f = st.selectbox("Fon", list(tbl.index), key="va_one")
            row_f = tbl.loc[sel_f].dropna()
            row_f = row_f[row_f > 0]
            if len(row_f):
                dn = pd.DataFrame({"Varlık": list(row_f.index),
                                   "Deger": row_f.values.astype(float)})
                st.altair_chart(donut_chart(dn, "Varlık", "Deger", th, height=300),
                                use_container_width=True, theme=None)
        st.dataframe(tbl.round(1), use_container_width=True)
        st.caption("Oranlar fon portföyündeki kategori payları olarak alınır; toplamın %100 "
                   "olması gerekmez.")

with T["Fon büyüklüğü ve akış"]:
    st.subheader("Fon büyüklüğü ve akış")
    if not live:
        st.info("Demo modunda TEFAS fon bilgisi yok. Canlı TEFAS verisini aç.")
    else:
        try:
            cur_i, ref_i, cur_d, ref_d = load_fund_info(str(pd.Timestamp.today().date()))
            ft = fund_table(cur_i, ref_i)
            ct = category_table(ft)
            st.caption(f"Kaynak: TEFAS fon bilgisi | Veri tarihi: {cur_d}"
                       + (f" | Karşılaştırma: {ref_d}" if ref_d else "")
                       + " | Kategoriler fon adından yaklaşık belirlenmiştir.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Piyasadaki fon", f"{len(ft):,}")
            k2.metric("Toplam portföy", f"{ft['Portföy büyüklüğü (TL)'].sum() / 1e9:,.1f} mlr TL")
            k3.metric("Toplam yatırımcı", f"{ft['Yatırımcı sayısı'].sum():,.0f}")
            fl_all = ft["Net akış (TL, tahmini)"].sum() if "Net akış (TL, tahmini)" in ft else np.nan
            k4.metric("Net akış (tahmini)", "-" if pd.isna(fl_all) else
                      f"{fl_all / 1e6:+,.1f} mn TL")

            st.markdown("##### Fon seçimi")
            pool = list(ft.sort_values("Portföy büyüklüğü (TL)", ascending=False).index)

            def lab(c):
                return f"{c} · {str(ft.loc[c, 'Fon adı'])[:55]}"
            s1, s2 = st.columns([3, 2])
            picked = s1.multiselect("Fon kodu veya adı ile ekle", pool, default=pool[:10],
                                    format_func=lab, key="fb_pick")
            cats_pick = s2.multiselect("Kategoriye göre ekle (tüm fonları)", list(ct.index),
                                       key="fb_cat")
            by_cat = ft[ft["Kategori"].isin(cats_pick)].sort_values(
                "Portföy büyüklüğü (TL)", ascending=False).index.tolist()
            codes = list(dict.fromkeys(list(picked) + by_cat))

            if not codes:
                st.info("Fon seç ya da bir kategori ekle.")
            else:
                sm = selection_summary(ft, codes)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Seçili fon", f"{sm['n']}")
                m2.metric("Seçili portföy", f"{sm['size'] / 1e9:,.2f} mlr TL")
                m3.metric("Piyasa payı", f"{sm['market_share']:.1%}")
                m4.metric("Seçili yatırımcı", f"{sm['investors']:,.0f}")
                tbl = sm["table"].copy()
                tbl["Seçim içi payı (%)"] = tbl["Portföy büyüklüğü (TL)"] / sm["size"]

                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("##### Seçili fonların pazar payı")
                    dn = pd.DataFrame({"Kod": tbl.index,
                                       "Deger": tbl["Portföy büyüklüğü (TL)"].values.astype(float)})
                    st.altair_chart(donut_chart(dn, "Kod", "Deger", th, height=320),
                                    use_container_width=True, theme=None)
                with g2:
                    st.markdown("##### Net akış (TL, tahmini)")
                    if "Net akış (TL, tahmini)" in tbl.columns:
                        pdf = pd.DataFrame({"Fon": tbl.index,
                                            "Akis": tbl["Net akış (TL, tahmini)"].astype(float).values})
                        st.altair_chart(bar_pos_neg(pdf, "Fon", "Akis", "", th, ".2s", height=320),
                                        use_container_width=True, theme=None)
                    else:
                        st.info("Net akış için iki tarihli veri gelmedi.")

                st.markdown("##### Seçili fonlar")
                fcols = [c for c in ["Fon adı", "Kategori", "Portföy büyüklüğü (TL)",
                                     "Pazar payı (%)", "Seçim içi payı (%)", "Yatırımcı sayısı",
                                     "Yatırımcı değişimi", "Büyüklük değişimi (%)",
                                     "Net akış (TL, tahmini)"] if c in tbl.columns]
                tfmt = {"Portföy büyüklüğü (TL)": "{:,.0f}", "Pazar payı (%)": "{:.2%}",
                        "Seçim içi payı (%)": "{:.1%}", "Yatırımcı sayısı": "{:,.0f}",
                        "Yatırımcı değişimi": "{:+,.0f}", "Büyüklük değişimi (%)": "{:+.2%}",
                        "Net akış (TL, tahmini)": "{:,.0f}"}
                st.dataframe(tbl[fcols].sort_values("Portföy büyüklüğü (TL)", ascending=False)
                             .style.format({k: v for k, v in tfmt.items() if k in fcols}),
                             use_container_width=True)
                st.download_button("Seçili fonları indir (CSV)",
                                   tbl[fcols].to_csv().encode("utf-8-sig"),
                                   "secili_fonlar.csv", "text/csv")

            with st.expander("Piyasa geneli: kategoriler"):
                cfmt = {"Portföy büyüklüğü (TL)": "{:,.0f}", "Yatırımcı sayısı": "{:,.0f}",
                        "Pazar payı (%)": "{:.2%}", "Yatırımcı payı (%)": "{:.2%}",
                        "Büyüklük değişimi (%)": "{:+.2%}", "Yatırımcı değişimi": "{:+,.0f}",
                        "Net akış (TL, tahmini)": "{:,.0f}"}
                st.dataframe(ct.style.format({k: v for k, v in cfmt.items() if k in ct.columns}),
                             use_container_width=True)
                dn2 = ct.reset_index()[["Kategori", "Portföy büyüklüğü (TL)"]]
                dn2.columns = ["Kategori", "Deger"]
                st.altair_chart(donut_chart(dn2, "Kategori", "Deger", th, height=300),
                                use_container_width=True, theme=None)
        except Exception as e:
            st.error(f"Fon bilgileri alınamadı: {e}. requirements.txt dosyasında 'pytefas' "
                     "olmalı.")

with T["Yatırımcı karşılaştırma"]:
    st.subheader("Yatırımcı karşılaştırma")
    st.caption("Fonları seç, tutarı gir. Geçmiş verilere göre stopaj ve masraflar "
               "düşüldükten sonra hangi fonun daha fazla getiri sağladığını ve hangisinin "
               "daha az risk taşıdığını yazılı verir.")
    pick = st.multiselect("Karşılaştırılacak fonlar", list(p.columns),
                          default=list(p.columns)[:3])
    amount = st.number_input("Yatırım tutarı (TL)", min_value=1000.0,
                             value=100000.0, step=1000.0)
    per = st.radio("Geçmiş dönem", list(PERIODS), index=3, horizontal=True)
    titles = {}
    if "title" in raw.columns:
        titles = (raw.dropna(subset=["title"]).drop_duplicates("code")
                  .set_index("code")["title"].to_dict())
    rates = {}
    with st.expander("Stopaj ve masraf ayarları"):
        st.caption("Stopaj oranı fon adından otomatik tahmin edilir (hisse senedi yoğun "
                   "fonlar %0, diğerleri %17,5 gerçek kişi için). Mevzuat sık değiştiği "
                   "için oranları fon izahnamesinden teyit edip değiştirebilirsin. "
                   "Fon yönetim ücreti fiyata dahildir, ayrıca düşülmez.")
        b1, b2 = st.columns(2)
        buy_fee = b1.number_input("Alım komisyonu (%)", 0.0, 10.0, 0.0, 0.05) / 100
        sell_fee = b2.number_input("Satım komisyonu (%)", 0.0, 10.0, 0.0, 0.05) / 100
        for c in pick:
            rate, kind = auto_stopaj(titles.get(c, ""))
            rates[c] = st.number_input(f"{c} stopaj oranı (%): {kind}", 0.0, 50.0,
                                       rate * 100, 0.5, key=f"stp_{c}") / 100
    if len(pick) < 2:
        st.info("Karşılaştırma için en az iki fon seç.")
    else:
        m = investor_metrics(p, pick, amount, PERIODS[per], rf, rates, buy_fee, sell_fee)
        if m.empty:
            st.warning("Seçilen fonlar için bu dönemde yeterli veri yok.")
        else:
            label = ("seçili veri aralığı" if PERIODS[per] is None
                     else f"son {per.lower()}")
            for para in investor_text(m, amount, label, rf):
                st.markdown(para)
            fmt = {"Brüt fon getirisi": "{:.2%}", "Brüt dönem sonu değer (TL)": "{:,.0f}",
                   "Brüt kâr/zarar (TL)": "{:,.0f}", "Stopaj oranı": "{:.2%}",
                   "Stopaj (TL)": "{:,.0f}", "Masraf (TL)": "{:,.0f}",
                   "Net elde edilen (TL)": "{:,.0f}", "Net kâr/zarar (TL)": "{:,.0f}",
                   "Net getiri": "{:.2%}", "Yıllık volatilite": "{:.2%}",
                   "Maks. düşüş": "{:.2%}", "Maks. düşüş (TL)": "{:,.0f}",
                   "Kötü ay %5 (getiri)": "{:.2%}", "Kötü ay %5 (TL)": "{:,.0f}",
                   "Sharpe": "{:.2f}"}
            st.dataframe(m.style.format(fmt), use_container_width=True)
            st.caption("Stopaj, dönem sonunda satış varsayımıyla yalnızca kazançtan "
                       "hesaplanır; zararda stopaj yoktur. Kötü ay %5: geçmişteki 21 "
                       "gözlemlik (yaklaşık 1 aylık) getirilerin en kötü %5'lik diliminin "
                       "eşiği. Risk göstergeleri brüt fiyat serisine dayanır.")

with T["ETF"]:
    render_etf(globals())
with T["ETF Analiz"]:
    render_etf_analiz(globals())
with T["ETF Detay"]:
    render_etf_detay(globals())

st.divider()
st.caption("Bu panel bilgilendirme amaçlıdır, yatırım tavsiyesi değildir. "
           "Geçmiş performans gelecekteki getirinin garantisi değildir. "
           "Veriler TEFAS kaynaklıdır; kritik rakamları resmi kaynaktan teyit ediniz.")
