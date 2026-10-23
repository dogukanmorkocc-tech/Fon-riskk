"""TEFAS Fon Analiz Paneli v3 - tutarlı getiri/risk hesaplaması
Çalıştır: streamlit run app.py   (requirements.txt aynı kalabilir)
"""
import numpy as np
import pandas as pd
import streamlit as st

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
    df.insert(0, "Sıra (3 Ay)", df["3 Ay"].rank(ascending=False).astype("Int64"))
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


def consistency_checks(p):
    rows = {}
    for c in p:
        s = p[c].dropna()
        if len(s) < 3:
            continue
        r = s.pct_change().dropna()
        total = s.iloc[-1] / s.iloc[0] - 1
        chain = (1 + r).prod() - 1
        logv = np.expm1(np.log1p(r).sum())
        days = (s.index[-1] - s.index[0]).days
        gaps = s.index.to_series().diff().dt.days.dropna()
        maxgap = int(gaps.max())
        expected = len(pd.bdate_range(s.index[0], s.index[-1]))
        big = r.abs().idxmax()
        diff = max(abs(total - chain), abs(total - logv))
        status = "Tutarlı" if diff < 1e-9 else "Kontrol et"
        if maxgap > 10:
            status += " | veri boşluğu"
        rows[c] = {
            "Durum": status,
            "Fiyattan dönem getirisi": total,
            "Zincirleme getiri": chain,
            "Log toplamından getiri": logv,
            "Fark": diff,
            "Yıllık (takvim günü)": (1 + total) ** (365.25 / days) - 1,
            "Yıllık (252 gün varsayımı)": (1 + total) ** (TD / len(r)) - 1,
            "Gözlem": len(s),
            "Eksik iş günü": expected - len(s),
            "En uzun boşluk (gün)": maxgap,
            "En büyük günlük hareket": r.loc[big],
            "Tarihi": big.strftime("%d.%m.%Y"),
        }
    return pd.DataFrame(rows).T


def flow_table(w):
    """Fon büyüklüğü, yatırımcı sayısı, tahmini net akış (Δpay x fiyat)."""
    if "number_of_shares" not in w:
        return None, None
    sh = w["number_of_shares"].ffill()
    pr = w["price"].ffill()
    flow = sh.diff().mul(pr)
    last = flow.index[-1]
    m1 = last - pd.DateOffset(months=1)
    aum = w["market_cap"].ffill() if "market_cap" in w else sh * pr
    rows = {}
    for c in pr:
        a = aum[c].dropna()
        row = {"Fon büyüklüğü (TL)": a.iloc[-1],
               "Büyüklük değişimi (1 Ay)": a.iloc[-1] / a.asof(m1) - 1
               if pd.notna(a.asof(m1)) else np.nan,
               "Net akış (1 Ay, TL)": flow.loc[flow.index > m1, c].sum()}
        if "number_of_investors" in w:
            inv = w["number_of_investors"][c].dropna()
            row["Yatırımcı sayısı"] = inv.iloc[-1]
            row["Yatırımcı değişimi (1 Ay)"] = inv.iloc[-1] / inv.asof(m1) - 1 \
                if pd.notna(inv.asof(m1)) else np.nan
        rows[c] = row
    return pd.DataFrame(rows).T, flow.cumsum()


def detect_anomalies(p, window=60, z=3.0):
    R = returns_df(p)
    zs = (R - R.rolling(window, min_periods=30).mean()) / \
        R.rolling(window, min_periods=30).std()
    out = zs.stack().rename("z").reset_index()
    out.columns = ["Tarih", "Fon", "z"]
    out["Getiri"] = [R.loc[d, f] for d, f in zip(out["Tarih"], out["Fon"])]
    return out[out["z"].abs() >= z].sort_values("Tarih", ascending=False)


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
    bg="#000000", card="#0a0a0a", card2="#161616", sidebar="#050505", border="#262626",
    text="#ffffff", muted="#a3a3a3", accent="#60a5fa", up="#34d399", down="#f87171",
    upbg="rgba(52,211,153,0.15)", downbg="rgba(248,113,113,0.15)", flatbg="#1a1a1a",
    palette=["#60a5fa", "#2dd4bf", "#fbbf24", "#a78bfa", "#f472b6", "#34d399",
             "#fb923c", "#22d3ee"],
    font="'Inter','Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif",
    chartfont="Inter", radius="10px", cr=3,
    hdrbg="linear-gradient(135deg,#0a0a0a 0%,#141414 100%)")

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
def basket_analysis(p, amounts, offset, rates, buy_fee=0.0, sell_fee=0.0, alpha=0.95):
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
    net_in = amt * (1 - buy_fee)
    gross = net_in * (1 + fret)
    gain = gross - net_in
    rate = pd.Series({c: rates.get(c, DEFAULT_STOPAJ) for c in funds})
    stopaj = gain.clip(lower=0) * rate           # stopaj yalnızca kazançtan, fon bazında
    sell = gross * sell_fee
    net_val = gross - stopaj - sell
    table = pd.DataFrame({
        "Tutar (TL)": amt, "Ağırlık": w, "Dönem getirisi": fret,
        "Risk katkısı": pd.Series(rc, index=funds),
        "Brüt değer (TL)": gross, "Stopaj (TL)": stopaj,
        "Masraf (TL)": amt * buy_fee + sell, "Net değer (TL)": net_val,
        "Net kâr/zarar (TL)": net_val - amt})
    return {"table": table, "value": value, "total": total,
            "ret": total.iloc[-1] / total.iloc[0] - 1, "vol": vol, "dd": dd,
            "var": var, "cvar": cvar, "div": div, "start": sub.index[0],
            "alpha": alpha}


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
UNIV_COLS = [
    ["code", "date", "price", "market_cap", "number_of_investors", "title"],
    ["code", "date", "price", "title"],
    ["code", "date", "price"],
]
PERIOD_DAYS = {"Son 1 hafta": 7, "Son 2 hafta": 14, "Son 1 ay": 30}
KIND_LABELS = {"YAT": "Yatırım fonları", "EMK": "Emeklilik fonları",
               "BYF": "Borsa yatırım fonları"}
BUL_FMT = {"Fiyat": "{:,.4f}", "Getiri": "{:.2%}", "Fon büyüklüğü (TL)": "{:,.0f}",
           "Net akış (TL)": "{:,.0f}", "Yatırımcı": "{:,.0f}",
           "Yatırımcı değişimi": "{:+,.0f}", "Yatırımcı değişimi (%)": "{:.2%}"}


def trnum(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pcn(x):
    return "-" if pd.isna(x) else pc(x)


def _fetch_day(crawler, d, kind, errs):
    """Bir iş gününde tüm fonların verisi; veri yoksa None."""
    if d.weekday() >= 5:
        return None
    ds = d.strftime("%Y-%m-%d")
    for cols in UNIV_COLS:
        try:
            kw = dict(start=ds, end=ds, columns=cols)
            if kind != "YAT":
                kw["kind"] = kind
            df = crawler.fetch(**kw)
        except Exception as e:
            errs.append(str(e))
            continue
        return df if df is not None and len(df) else None
    return None


@st.cache_data(show_spinner="TEFAS tüm fon verisi çekiliyor (1-2 dakika sürebilir)...",
               ttl=3600)
def load_universe(kind, end_str, days):
    from tefas import Crawler
    crawler = Crawler()
    errs, end = [], pd.Timestamp(end_str)
    df_last = last_d = None
    for back in range(8):
        d = end - pd.Timedelta(days=back)
        df_last = _fetch_day(crawler, d, kind, errs)
        if df_last is not None:
            last_d = d
            break
    if df_last is None:
        raise RuntimeError("TEFAS'tan tüm fon verisi alınamadı. "
                           + (errs[-1] if errs else "Son günlerde veri bulunamadı."))
    target = last_d - pd.Timedelta(days=days)
    df_ref = ref_d = None
    for back in range(8):
        d = target - pd.Timedelta(days=back)
        df_ref = _fetch_day(crawler, d, kind, errs)
        if df_ref is not None:
            ref_d = d
            break
    if df_ref is None:
        raise RuntimeError("Karşılaştırma tarihi için TEFAS verisi alınamadı.")
    return df_last, df_ref, last_d, ref_d


def synthetic_universe(days, end_str, n=80, seed=7):
    """Demo modu için sentetik fon evreni (iki tarih)."""
    rng = np.random.default_rng(seed)
    last_d = pd.Timestamp(end_str)
    while last_d.weekday() >= 5:
        last_d -= pd.Timedelta(days=1)
    ref_d = last_d - pd.Timedelta(days=days)
    while ref_d.weekday() >= 5:
        ref_d -= pd.Timedelta(days=1)
    codes = [f"F{i:03d}" for i in range(1, n + 1)]
    p0, ret = rng.uniform(1, 200, n), rng.normal(0.004, 0.03, n)
    sh0, sh_g = rng.lognormal(15, 1.2, n), rng.normal(0.01, 0.08, n)
    inv0, inv_g = rng.integers(50, 80000, n).astype(float), rng.normal(0.01, 0.06, n)

    def mk(d, p, sh, inv):
        return pd.DataFrame({"code": codes, "date": d, "price": p, "market_cap": p * sh,
                             "number_of_investors": inv.round(),
                             "title": [f"DEMO FON {c}" for c in codes]})
    return (mk(last_d, p0 * (1 + ret), sh0 * (1 + sh_g), inv0 * (1 + inv_g)),
            mk(ref_d, p0, sh0, inv0), last_d, ref_d)


def bulletin_frame(df_last, df_ref):
    """İki tarih arasındaki getiri, yatırımcı değişimi ve tahmini net akış."""
    def prep(df):
        d = df.copy()
        for c in ["price", "market_cap", "number_of_investors"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d.drop_duplicates("code").set_index("code")
    a, b = prep(df_last), prep(df_ref)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    out = pd.DataFrame(index=idx)
    out.index.name = "Fon"
    out["Fon adı"] = a["title"] if "title" in a.columns else ""
    out["Fiyat"] = a["price"]
    out["Getiri"] = (a["price"] / b["price"] - 1).where(b["price"] > 0)
    if "market_cap" in a.columns and "market_cap" in b.columns:
        out["Fon büyüklüğü (TL)"] = a["market_cap"]
        sh_a, sh_b = a["market_cap"] / a["price"], b["market_cap"] / b["price"]
        out["Net akış (TL)"] = (sh_a - sh_b) * a["price"]
    if "number_of_investors" in a.columns and "number_of_investors" in b.columns:
        ia, ib = a["number_of_investors"], b["number_of_investors"]
        out["Yatırımcı"] = ia
        out["Yatırımcı değişimi"] = ia - ib
        out["Yatırımcı değişimi (%)"] = (ia / ib - 1).where(ib > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def apply_filters(f, min_mc, min_inv):
    m = pd.Series(True, index=f.index)
    if "Fon büyüklüğü (TL)" in f.columns:
        m &= f["Fon büyüklüğü (TL)"].fillna(0) >= min_mc
    if "Yatırımcı" in f.columns:
        m &= f["Yatırımcı"].fillna(0) >= min_inv
    return f[m]


def top_n(f, col, n, ascending=False):
    if col not in f.columns:
        return f.iloc[0:0]
    g = f.dropna(subset=[col])
    g = g[g[col] < 0] if ascending else g[g[col] > 0]   # yükselen/giriş > 0, düşen/çıkış < 0
    return g.sort_values(col, ascending=ascending).head(n)


def bulletin_text(f, last_d, ref_d, n, min_mc, min_inv):
    """Kopyalanabilir bülten metni."""
    def nm(code):
        name = str(f.loc[code, "Fon adı"]) if "Fon adı" in f.columns else ""
        return f"{code} - {name}" if name and name != "nan" else str(code)

    def sgn(x, suffix=""):
        return ("+" if x >= 0 else "-") + trnum(abs(x)) + suffix

    L = [f"TEFAS Fon Bülteni | {ref_d:%d.%m.%Y} - {last_d:%d.%m.%Y}", ""]

    def block(title, df, fmt):
        if df.empty:
            return
        L.append(title)
        for i, (code, r) in enumerate(df.iterrows(), 1):
            L.append(f"{i}. {nm(code)}: {fmt(r)}")
        L.append("")

    block("En çok yükselen fonlar (fiyat getirisi)", top_n(f, "Getiri", n),
          lambda r: pcn(r["Getiri"]))
    block("En çok düşen fonlar (fiyat getirisi)", top_n(f, "Getiri", n, True),
          lambda r: pcn(r["Getiri"]))
    inv_fmt = lambda r: f"{sgn(r['Yatırımcı değişimi'])} kişi ({pcn(r['Yatırımcı değişimi (%)'])})"
    block("Yatırımcı sayısı en çok artan fonlar (giriş)",
          top_n(f, "Yatırımcı değişimi", n), inv_fmt)
    block("Yatırımcı sayısı en çok azalan fonlar (çıkış)",
          top_n(f, "Yatırımcı değişimi", n, True), inv_fmt)
    flow_fmt = lambda r: ("+" if r["Net akış (TL)"] >= 0 else "-") + \
        trnum(abs(r["Net akış (TL)"]) / 1e6, 1) + " mn TL"
    block("Tahmini net para girişi en yüksek fonlar", top_n(f, "Net akış (TL)", n), flow_fmt)
    block("Tahmini net para çıkışı en yüksek fonlar",
          top_n(f, "Net akış (TL)", n, True), flow_fmt)
    L.append(f"Kapsam: {len(f)} fon (fon büyüklüğü en az {tl(min_mc)}, "
             f"yatırımcı sayısı en az {trnum(min_inv)}).")
    L.append("Kaynak: TEFAS. Net akış, pay sayısı değişimi x fiyat ile tahmin edilmiştir. "
             "Geçmiş performans gelecekteki getirinin garantisi değildir; metin "
             "bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.")
    return "\n".join(L)


def show_bul_table(df, cols):
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols].style.format({k: v for k, v in BUL_FMT.items() if k in cols}),
                 use_container_width=True)


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


def perf_summary(p, bench):
    """Günlük istatistikler ve kıyasa göre yukarı/aşağı yakalama oranları."""
    R = returns_df(p)
    has_b = bench in R.columns
    rows = {}
    for c in R.columns:
        r = R[c].dropna()
        if len(r) < 10:
            continue
        row = {"Gözlem": len(r), "Pozitif gün %": (r > 0).mean(),
               "En iyi gün": r.max(), "En kötü gün": r.min(),
               "Ort. yükseliş günü": r[r > 0].mean(), "Ort. düşüş günü": r[r < 0].mean(),
               "Yukarı yakalama": np.nan, "Aşağı yakalama": np.nan}
        if has_b and c != bench:
            pair = pd.concat([R[c], R[bench]], axis=1).dropna()
            up = pair[pair[bench] > 0]
            dn = pair[pair[bench] < 0]
            if len(up) > 5 and up[bench].mean() != 0:
                row["Yukarı yakalama"] = up[c].mean() / up[bench].mean()
            if len(dn) > 5 and dn[bench].mean() != 0:
                row["Aşağı yakalama"] = dn[c].mean() / dn[bench].mean()
        rows[c] = row
    return pd.DataFrame(rows).T


# ---------- Piyasalar (Yahoo Finance) ----------
YF_GROUPS = {
    "Tahvil ve faiz": [("^TNX", "ABD 10 yıllık Hazine"), ("^FVX", "ABD 5 yıllık Hazine"),
                       ("^IRX", "ABD 13 haftalık bono")],
    "Kur": [("USDTRY=X", "USD/TRY"), ("EURTRY=X", "EUR/TRY"), ("EURUSD=X", "EUR/USD")],
    "Endeks": [("XU100.IS", "BIST 100"), ("XU030.IS", "BIST 30"),
               ("^GSPC", "S&P 500"), ("^VIX", "VIX")],
    "BIST hisse": [("THYAO.IS", "Türk Hava Yolları"), ("GARAN.IS", "Garanti BBVA"),
                   ("AKBNK.IS", "Akbank"), ("ISCTR.IS", "İş Bankası (C)"),
                   ("BIMAS.IS", "BİM"), ("SISE.IS", "Şişecam")],
    "Emtia": [("GC=F", "Altın (ons)"), ("SI=F", "Gümüş (ons)"), ("BZ=F", "Brent petrol")],
}
YF_NAME = {s: n for g in YF_GROUPS.values() for s, n in g}
YF_GROUP_OF = {s: g for g, items in YF_GROUPS.items() for s, _ in items}
YIELD_GROUP = "Tahvil ve faiz"
# Yahoo'nun ABD faiz endeksleri değeri yüzde × 10 olarak verir (ör. 42,5 = %4,25)
YF_SCALE = {"^TNX": 0.1, "^FVX": 0.1, "^IRX": 0.1}


@st.cache_data(ttl=900, show_spinner="Yahoo Finance'ten canlı veri çekiliyor...")
def load_yahoo(symbols, period):
    import yfinance as yf
    syms = list(symbols)
    df = yf.download(syms, period=period, interval="1d", auto_adjust=True,
                     progress=False, threads=True)
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance boş veri döndürdü")
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]].rename(columns={"Close": syms[0]})
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    for s, k in YF_SCALE.items():
        if s in close.columns:
            close[s] = close[s] * k
    return close.dropna(how="all")


def demo_yahoo(symbols, days=400, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    out = {}
    for s in symbols:
        if YF_GROUP_OF.get(s) == YIELD_GROUP:
            base, sg = rng.uniform(3, 45), 0.02
        elif "=X" in s:
            base, sg = rng.uniform(1, 40), 0.004
        else:
            base, sg = rng.uniform(5, 300), 0.012
        out[s] = base * np.exp(np.cumsum(rng.normal(0, sg, len(idx))))
    return pd.DataFrame(out, index=idx)


def snapshot(close):
    """Son değer ile 1G / 1H / 1A / YBB değişimi: yüzde ve fark (faizler için)."""
    last = close.index[-1]
    refs = {"1G": None, "1H": last - pd.Timedelta(days=7),
            "1A": last - pd.DateOffset(months=1),
            "YBB": pd.Timestamp(year=last.year - 1, month=12, day=31)}
    rows = {}
    for c in close.columns:
        s = close[c].dropna()
        if s.empty:
            continue
        v = float(s.iloc[-1])
        row = {"Ad": YF_NAME.get(c, c), "Grup": YF_GROUP_OF.get(c, ""), "Son": v}
        for k, t in refs.items():
            if k == "1G":
                ref = float(s.iloc[-2]) if len(s) > 1 else np.nan
            else:
                ref = s.asof(t)
                ref = float(ref) if pd.notna(ref) else np.nan
            row[k] = (v / ref - 1) if (pd.notna(ref) and ref != 0) else np.nan
            row[k + "_d"] = (v - ref) if pd.notna(ref) else np.nan
        rows[c] = row
    return pd.DataFrame(rows).T


def yahoo_table_html(snap, th):
    muted, border = th["muted"], th["border"]

    def chg(row, k, yld):
        if yld:
            x = row[k + "_d"]
            txt = "" if pd.isna(x) else f"{x * 100:+.0f} bp"
        else:
            x = row[k]
            txt = "" if pd.isna(x) else f"{x * 100:+.2f}%"
        if pd.isna(x):
            return (f"<td style='text-align:right;color:{muted};"
                    f"border-bottom:1px solid {border}'>–</td>")
        col = th["up"] if x >= 0 else th["down"]
        return (f"<td style='text-align:right;color:{col};font-weight:600;"
                f"border-bottom:1px solid {border}'>{txt.replace('.', ',')}</td>")

    th_style = (f"padding:8px 10px;color:{muted};font-weight:600;font-size:12px;"
                f"border-bottom:1px solid {border};")
    head = ("<tr>"
            f"<th style='{th_style}text-align:left'>Sembol / ad</th>"
            f"<th style='{th_style}text-align:right'>Son</th>"
            f"<th style='{th_style}text-align:right'>1G</th>"
            f"<th style='{th_style}text-align:right'>1H</th>"
            f"<th style='{th_style}text-align:right'>1A</th>"
            f"<th style='{th_style}text-align:right'>YBB</th></tr>")
    body = []
    order = [s for g in YF_GROUPS.values() for s, _ in g]
    snap = snap.reindex([s for s in order if s in snap.index])
    for sym, r in snap.iterrows():
        yld = r["Grup"] == YIELD_GROUP
        last = (f"%{trnum(r['Son'], 2)}" if yld else trnum(r["Son"], 2))
        body.append(
            "<tr>"
            f"<td style='padding:9px 10px;border-bottom:1px solid {border}'>"
            f"<div style='font-weight:600'>{r['Ad']}</div>"
            f"<div style='font-size:11px;color:{muted}'>{sym} · {r['Grup']}</div></td>"
            f"<td style='text-align:right;font-weight:600;border-bottom:1px solid {border}'>{last}</td>"
            + "".join(chg(r, k, yld) for k in ["1G", "1H", "1A", "YBB"])
            + "</tr>")
    return ("<div style='overflow-x:auto'><table style='width:100%;border-collapse:collapse;"
            "font-size:13px;font-variant-numeric:tabular-nums'>"
            f"{head}{''.join(body)}</table></div>")


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


# ---------- Arayüz ----------
with st.sidebar:
    st.header("Ayarlar")
    title = st.text_input("Panel başlığı", "TEFAS Fon Analiz Paneli")
    tema = st.selectbox("Görünüm", list(THEMES))
    codes = [c.strip().upper() for c in
             st.text_input("Fon kodları (virgülle)", "AFA,TI2,IPB,YAS").split(",")
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
NAMES = ['Home', 'Piyasalar', 'Haftalık bülten', 'Portföy sepeti', 'Yatırımcı karşılaştırma', 'Özet', 'Performans', 'Risk', 'Varlık dağılımı', 'Fon büyüklüğü ve akış', 'Anomaliler']
T = dict(zip(NAMES, st.tabs(NAMES)))

with T["Home"]:
    st.markdown(ticker_html(p), unsafe_allow_html=True)
    ft_h, _ = flow_table(w)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fon sayısı", len(p.columns))
    if "market_cap" in w:
        k2.metric("Toplam fon büyüklüğü",
                  f"{w['market_cap'].ffill().iloc[-1].sum() / 1e9:,.2f} mlr TL")
    else:
        k2.metric("Toplam fon büyüklüğü", "-")
    if "number_of_investors" in w:
        k3.metric("Toplam yatırımcı",
                  f"{w['number_of_investors'].ffill().iloc[-1].sum():,.0f}")
    else:
        k3.metric("Toplam yatırımcı", "-")
    if ft_h is not None:
        k4.metric("1 ay net akış (tahmini)",
                  f"{ft_h['Net akış (1 Ay, TL)'].astype(float).sum() / 1e6:,.1f} mn TL")
    else:
        k4.metric("1 ay net akış (tahmini)", "-")

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
        st.markdown("##### 1 aylık tahmini net akış (TL)")
        if ft_h is not None:
            nf = ft_h["Net akış (1 Ay, TL)"].astype(float).rename("Akış").rename_axis("Fon").reset_index()
            st.altair_chart(bar_pos_neg(nf, "Fon", "Akış", "", th, ".2s"),
                            use_container_width=True, theme=None)
        else:
            st.info("Fon büyüklüğü verisi gelmedi, akış grafiği gösterilemiyor.")

    if "market_cap" in w or "number_of_investors" in w:
        h5, h6 = st.columns(2)
        with h5:
            if "market_cap" in w:
                st.markdown("##### Fon büyüklüğü dağılımı")
                mc = (w["market_cap"].ffill().iloc[-1].dropna().astype(float)
                      .rename("Deger").rename_axis("Fon").reset_index())
                st.altair_chart(donut_chart(mc, "Fon", "Deger", th),
                                use_container_width=True, theme=None)
        with h6:
            if "number_of_investors" in w:
                st.markdown("##### Yatırımcı sayısı dağılımı")
                iv = (w["number_of_investors"].ffill().iloc[-1].dropna().astype(float)
                      .rename("Deger").rename_axis("Fon").reset_index())
                st.altair_chart(donut_chart(iv, "Fon", "Deger", th),
                                use_container_width=True, theme=None)

with T["Piyasalar"]:
    st.subheader("Piyasalar")
    st.caption("Tahvil faizleri, kurlar, endeksler, BIST hisseleri ve emtia. Canlı modda "
               "Yahoo Finance'ten çekilir, 15 dakika önbelleklenir. Faizlerde değişim baz "
               "puan (bp) olarak gösterilir.")
    pz_all = [s for g in YF_GROUPS.values() for s, _ in g]
    pz_per = st.selectbox("Geçmiş", ["6mo", "1y", "2y", "5y"], index=1, key="pz_per")
    pz_close = None
    try:
        pz_close = (load_yahoo(tuple(pz_all), pz_per) if live
                    else demo_yahoo(pz_all, seed=11))
    except Exception as e:
        st.error(f"Piyasa verisi alınamadı: {e}. Yahoo Finance kullanımı için "
                 "requirements.txt dosyasına 'yfinance' satırı eklenmiş olmalı.")
    if pz_close is not None and not pz_close.empty:
        missing = [s for s in pz_all if s not in pz_close.columns or pz_close[s].dropna().empty]
        if missing:
            st.caption("Veri gelmeyen semboller: " + ", ".join(missing) + ".")
        snap = snapshot(pz_close)
        st.markdown(yahoo_table_html(snap, th), unsafe_allow_html=True)

        st.markdown("##### Grafik")
        avail = [s for s in pz_all if s in pz_close.columns]
        default_sel = [s for s in ["XU100.IS", "USDTRY=X", "GC=F"] if s in avail]
        g1, g2, g3 = st.columns([3, 1, 1])
        pz_sel = g1.multiselect("Karşılaştır", avail, default=default_sel, key="pz_sel",
                                format_func=lambda s: f"{YF_NAME.get(s, s)} ({s})")
        pz_norm = g2.checkbox("Normalize (başlangıç = 100)", value=True, key="pz_norm")
        pz_rng = g3.radio("Aralık", ["1A", "6A", "Tümü"], index=0, horizontal=True,
                          key="pz_rng")
        if pz_sel:
            d = pz_close[pz_sel].dropna(how="all")
            if pz_rng != "Tümü":
                months = 1 if pz_rng == "1A" else 6
                d = d[d.index >= d.index[-1] - pd.DateOffset(months=months)]
            d = d.dropna(how="all").ffill()
            if pz_norm:
                d = d / d.apply(lambda s: s.dropna().iloc[0]) * 100
            d = d.rename(columns=lambda s: YF_NAME.get(s, s))
            st.altair_chart(line_chart(d, th, "", ",.2f", height=280),
                            use_container_width=True, theme=None)
            if pz_norm and any(YF_GROUP_OF.get(s) == YIELD_GROUP for s in pz_sel):
                st.caption("Not: faiz serilerini normalize etmek getiri ekseninde yanıltıcı "
                           "olabilir; faiz düzeyini görmek için kutucuğu kaldır.")
        st.markdown("##### Repo ve politika faizi")
        st.number_input("TCMB politika faizi (%), manuel", 0.0, 100.0, 40.0, 0.25, key="pz_pol",
                        help="Yahoo Finance'te TCMB politika faizi veya repo ihalesi "
                             "bulunmuyor. Güncel oranı TCMB duyurusundan gir. 40 yer tutucudur.")
        st.caption("Repo ve politika faizi Yahoo Finance'te yer almadığı için manuel girilir. "
                   "Bu oran kenar çubuğundaki risksiz oran olarak da kullanılabilir.")

with T["Haftalık bülten"]:
    st.subheader("Haftalık fon bülteni")
    st.caption("TEFAS'taki tüm fonlar taranır: en çok yükselen ve düşen fonlar, yatırımcı "
               "sayısı en çok artan (giriş) ve azalan (çıkış) fonlar, tahmini net akış.")
    bc1, bc2, bc3 = st.columns(3)
    bl_per = bc1.selectbox("Dönem", list(PERIOD_DAYS), key="bl_per")
    bl_kind = bc2.selectbox("Fon türü", list(KIND_LABELS),
                            format_func=lambda k: KIND_LABELS[k], key="bl_kind")
    bl_n = bc3.slider("Listedeki fon sayısı", 5, 20, 10, key="bl_n")
    with st.expander("Filtreler (çok küçük fonların listeyi bozmasını önler)"):
        f1, f2 = st.columns(2)
        bl_mc = f1.number_input("En az fon büyüklüğü (mn TL)", 0.0, 1e6, 10.0, 5.0,
                                key="bl_mc") * 1e6
        bl_inv = f2.number_input("En az yatırımcı sayısı", 0.0, 1e7, 100.0, 50.0,
                                 key="bl_inv")
    if st.button("Bülteni hazırla", type="primary", key="bl_btn"):
        st.session_state["bl_ready"] = True
    if not st.session_state.get("bl_ready"):
        st.info("Ayarları seçip **Bülteni hazırla** düğmesine dokun. Tüm fonların verisi "
                "çekileceği için ilk açılış biraz sürebilir.")
    else:
        bl_data, bl_err = None, None
        try:
            today_s = str(pd.Timestamp.today().date())
            bl_data = (load_universe(bl_kind, today_s, PERIOD_DAYS[bl_per]) if live
                       else synthetic_universe(PERIOD_DAYS[bl_per], today_s))
        except Exception as e:
            bl_err = str(e)
        if bl_err:
            st.error(f"Bülten verisi alınamadı: {bl_err}")
        else:
            df_last, df_ref, last_d, ref_d = bl_data
            bf_all = bulletin_frame(df_last, df_ref)
            bf = apply_filters(bf_all, bl_mc, bl_inv)
            st.caption(f"Karşılaştırma: {ref_d:%d.%m.%Y} → {last_d:%d.%m.%Y} | "
                       f"{'TEFAS (canlı)' if live else 'sentetik demo verisi'}")
            if bf.empty:
                st.warning("Filtrelere uyan fon yok. Filtreleri gevşetmeyi dene.")
            else:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Taranan fon", f"{len(bf_all):,}")
                q2.metric("Filtre sonrası", f"{len(bf):,}")
                q3.metric("Medyan getiri", pcn(bf["Getiri"].median()))
                up_share = (bf["Getiri"].dropna() > 0).mean()
                q4.metric("Yükselen fon oranı", pcn(up_share))
                sb = st.tabs(["Getiri", "Yatırımcı giriş/çıkış", "Net akış (TL)",
                              "Bülten metni"])
                with sb[0]:
                    up, dn = top_n(bf, "Getiri", bl_n), top_n(bf, "Getiri", bl_n, True)
                    mv = (pd.concat([up.head(5), dn.head(5)])[["Getiri"]]
                          .reset_index().drop_duplicates("Fon"))
                    st.markdown("##### En çok yükselen ve düşen 5'er fon")
                    st.altair_chart(bar_pos_neg(mv, "Fon", "Getiri", "", th),
                                    use_container_width=True, theme=None)
                    st.markdown("##### En çok yükselen fonlar")
                    show_bul_table(up, ["Fon adı", "Getiri", "Fiyat",
                                        "Fon büyüklüğü (TL)", "Yatırımcı"])
                    st.markdown("##### En çok düşen fonlar")
                    show_bul_table(dn, ["Fon adı", "Getiri", "Fiyat",
                                        "Fon büyüklüğü (TL)", "Yatırımcı"])
                with sb[1]:
                    if "Yatırımcı değişimi" not in bf.columns:
                        st.info("TEFAS'tan yatırımcı sayısı gelmedi.")
                    else:
                        ia = top_n(bf, "Yatırımcı değişimi", bl_n)
                        ib = top_n(bf, "Yatırımcı değişimi", bl_n, True)
                        mv2 = (pd.concat([ia.head(5), ib.head(5)])[["Yatırımcı değişimi"]]
                               .rename(columns={"Yatırımcı değişimi": "Degisim"})
                               .reset_index().drop_duplicates("Fon"))
                        st.markdown("##### Yatırımcı sayısı değişimi (kişi), 5 artan + 5 azalan")
                        st.altair_chart(bar_pos_neg(mv2, "Fon", "Degisim", "", th, ",.0f"),
                                        use_container_width=True, theme=None)
                        cols_i = ["Fon adı", "Yatırımcı", "Yatırımcı değişimi",
                                  "Yatırımcı değişimi (%)", "Getiri", "Net akış (TL)"]
                        st.markdown("##### Yatırımcı sayısı en çok artan fonlar (giriş)")
                        show_bul_table(ia, cols_i)
                        st.markdown("##### Yatırımcı sayısı en çok azalan fonlar (çıkış)")
                        show_bul_table(ib, cols_i)
                with sb[2]:
                    if "Net akış (TL)" not in bf.columns:
                        st.info("TEFAS'tan fon büyüklüğü gelmedi, net akış hesaplanamadı.")
                    else:
                        fa = top_n(bf, "Net akış (TL)", bl_n)
                        fb = top_n(bf, "Net akış (TL)", bl_n, True)
                        mv3 = (pd.concat([fa.head(5), fb.head(5)])[["Net akış (TL)"]]
                               .rename(columns={"Net akış (TL)": "Akis"})
                               .reset_index().drop_duplicates("Fon"))
                        st.markdown("##### Tahmini net akış (TL), 5 giriş + 5 çıkış")
                        st.altair_chart(bar_pos_neg(mv3, "Fon", "Akis", "", th, ".2s"),
                                        use_container_width=True, theme=None)
                        cols_f = ["Fon adı", "Net akış (TL)", "Fon büyüklüğü (TL)",
                                  "Getiri", "Yatırımcı değişimi"]
                        st.markdown("##### En yüksek net para girişi")
                        show_bul_table(fa, cols_f)
                        st.markdown("##### En yüksek net para çıkışı")
                        show_bul_table(fb, cols_f)
                        st.caption("Net akış = (pay sayısı değişimi) x fiyat; pay sayısı "
                                   "fon büyüklüğü / fiyat ile türetilir. Tahmindir.")
                with sb[3]:
                    txt = bulletin_text(bf, last_d, ref_d, min(bl_n, 10), bl_mc, bl_inv)
                    st.caption("Metni kopyala (sağ üstteki simge) veya indir; "
                               "LinkedIn/bülten taslağı olarak kullanılabilir.")
                    st.code(txt, language=None)
                    st.download_button("Bülten metnini indir (.txt)", txt.encode("utf-8"),
                                       "tefas_bulten.txt", "text/plain")
                    st.download_button("Tüm fonlar (CSV)",
                                       bf.reset_index().to_csv(index=False).encode("utf-8-sig"),
                                       "tefas_haftalik_fonlar.csv", "text/csv")

with T["Portföy sepeti"]:
    st.subheader("Portföy sepeti")
    st.caption("Her fona yatıracağın tutarı gir; dağılım, risk ve geçmiş performans "
               "birlikte gösterilir.")
    bk_pick = st.multiselect("Sepetteki fonlar", list(p.columns),
                             default=list(p.columns)[:3], key="bk_pick")
    bk_per = st.radio("Geçmiş dönem", list(PERIODS), index=3, horizontal=True, key="bk_per")
    amounts = {}
    if bk_pick:
        cols_b = st.columns(min(len(bk_pick), 4))
        for i, c in enumerate(bk_pick):
            amounts[c] = cols_b[i % len(cols_b)].number_input(
                f"{c} tutarı (TL)", 0.0, 1e12, 50000.0, 1000.0, key=f"bk_amt_{c}")
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
    if not bk_pick or sum(amounts.values()) <= 0:
        st.info("En az bir fon seç ve tutar gir.")
    else:
        res = basket_analysis(p, amounts, PERIODS[bk_per], rates_b, bk_buy, bk_sell, alpha)
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
                     "Net değer (TL)": "{:,.0f}", "Net kâr/zarar (TL)": "{:,.0f}"}
            st.dataframe(tb.style.format(fmt_b), use_container_width=True)
            st.download_button("Tabloyu indir (CSV)", tb.to_csv().encode("utf-8-sig"),
                               "sepet.csv", "text/csv")

with T["Özet"]:
    st.subheader("Dönemsel getiri ve sıralama")
    pr = period_returns(p)
    st.dataframe(pr.style.format({c: "{:.2%}" for c in pr.columns if c != "Sıra (3 Ay)"}),
                 use_container_width=True)

with T["Performans"]:
    st.subheader("Normalize fiyat (başlangıç = 100)")
    base = p.apply(lambda s: s.dropna().iloc[0])
    st.altair_chart(line_chart(pf / base * 100, th, "", ",.0f"),
                    use_container_width=True, theme=None)
    st.subheader("Zirveden düşüş (drawdown)")
    st.altair_chart(line_chart(pf / pf.cummax() - 1, th, "", ".0%"),
                    use_container_width=True, theme=None)

with T["Performans"]:
    st.subheader("Fon yöneticisi tabloları")
    st.caption("Kıyas fonuna göre göreli performans, aylık getiri ısı haritası, "
               "yakalama oranları ve günlük istatistikler.")
    rel = relative_table(p, bench)
    if rel is not None:
        st.markdown(f"##### Kıyasa göre fark (fon − {bench}), yüzde puan")
        st.dataframe(rel.style.format("{:+.2%}"), use_container_width=True)
    mr = monthly_returns(p)
    if len(mr):
        st.markdown("##### Aylık getiri ısı haritası")
        st.altair_chart(monthly_heatmap(mr, th), use_container_width=True, theme=None)
    st.markdown("##### Kayan 1 aylık getiri")
    st.altair_chart(line_chart(pf.pct_change(21).dropna(how="all"), th, "", ".0%"),
                    use_container_width=True, theme=None)
    ps = perf_summary(p, bench)
    if not ps.empty:
        st.markdown("##### Günlük istatistikler ve yakalama oranları")
        fmt_ps = {"Pozitif gün %": "{:.1%}", "En iyi gün": "{:.2%}", "En kötü gün": "{:.2%}",
                  "Ort. yükseliş günü": "{:.2%}", "Ort. düşüş günü": "{:.2%}",
                  "Yukarı yakalama": "{:.2f}", "Aşağı yakalama": "{:.2f}"}
        st.dataframe(ps.style.format({k: v for k, v in fmt_ps.items() if k in ps.columns}),
                     use_container_width=True)
        st.caption("Yukarı yakalama: kıyasın yükseliş günlerinde fonun ortalama getirisinin "
                   "kıyasa oranı; 1'in üzerinde ise fon yükselişte kıyastan güçlü. "
                   "Aşağı yakalama: düşüş günlerinde aynı oran; 1'in altı, fonun düşüşte "
                   "kıyasa göre daha az kaybettiğini gösterir.")

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

with T["Risk"]:
    rt = risk_table(p, bench, rf, alpha)
    fm = {c: "{:.2%}" for c in rt.columns if c != "Sharpe"}
    fm["Sharpe"] = "{:.2f}"
    st.dataframe(rt.style.format(fm), use_container_width=True)
    if (p.index[-1] - p.index[0]).days < 90:
        st.warning("Dönem 90 günden kısa: yıllıklandırılmış değerler yanıltıcı olabilir.")
    st.download_button("Tabloyu indir (CSV)", rt.to_csv().encode("utf-8-sig"),
                       "fon_risk.csv", "text/csv")
    st.subheader("Risk-getiri dağılımı")
    sc2 = (rt[["Yıllık volatilite", "Yıllık getiri"]].astype(float)
           .rename(columns={"Yıllık volatilite": "Vol", "Yıllık getiri": "Getiri"})
           .rename_axis("Fon").reset_index())
    st.altair_chart(scatter_chart(sc2, th, ""), use_container_width=True, theme=None)
    st.subheader("30 gözlemlik kayan volatilite (yıllıklandırılmış)")
    R = returns_df(p)
    rv = R.rolling(30, min_periods=20).std()
    rvv = rv * np.sqrt(pd.Series({c: obs_per_year(p[c]) for c in p}))
    st.altair_chart(line_chart(rvv, th, "", ".0%"), use_container_width=True, theme=None)
    st.subheader("Getiri korelasyonu (ortak tarihler)")
    st.altair_chart(corr_heatmap(R.corr(), th), use_container_width=True, theme=None)

with T["Fon büyüklüğü ve akış"]:
    ft, cum = flow_table(w)
    if ft is None:
        st.info("TEFAS'tan fon büyüklüğü bilgisi gelmedi. "
                f"Gelen sütunlar: {', '.join(map(str, raw.columns))}")
    else:
        fm = {"Fon büyüklüğü (TL)": "{:,.0f}", "Net akış (1 Ay, TL)": "{:,.0f}",
              "Büyüklük değişimi (1 Ay)": "{:.2%}", "Yatırımcı sayısı": "{:,.0f}",
              "Yatırımcı değişimi (1 Ay)": "{:.2%}"}
        st.dataframe(ft.style.format({k: v for k, v in fm.items() if k in ft.columns}),
                     use_container_width=True)
        st.caption("Net akış = pay sayısı değişimi x fiyat (tahmini).")
        st.subheader("Kümülatif tahmini net akış (TL)")
        st.altair_chart(line_chart(cum, th, "", ".2s"), use_container_width=True, theme=None)

with T["Anomaliler"]:
    st.caption("60 gözlemlik kayan pencerede |z| >= 3 olan günlük getiriler")
    st.dataframe(detect_anomalies(p), use_container_width=True)

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

st.divider()
st.caption("Bu panel bilgilendirme amaçlıdır, yatırım tavsiyesi değildir. "
           "Geçmiş performans gelecekteki getirinin garantisi değildir. "
           "Veriler TEFAS kaynaklıdır; kritik rakamları resmi kaynaktan teyit ediniz.")
