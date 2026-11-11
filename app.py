"""Portföy Analiz Paneli v5 - sade arayüz: profil anketi + profile uygun fon sepeti akışı
Çalıştır: streamlit run app.py   (requirements.txt aynı kalabilir)
"""
import numpy as np
import pandas as pd
import streamlit as st
from profil_panel import render_profil_form
from bulten_panel import render_bulten

st.set_page_config(page_title="Portföy Analiz Paneli", layout="wide")
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








# ---------- TEFAS risk değeri (1-7) ----------
# SPK Yatırımcı Bilgi Formu yöntemi: haftalık getirilerin yıllık standart sapması 7 banda çevrilir.
RISK_BANTLARI = [(0.5, 1), (2, 2), (5, 3), (10, 4), (15, 5), (25, 6)]


def risk_degeri(s):
    """TEFAS tarzı risk değeri. Dönüş: (1-7 arası değer, haftalık getirilerden yıllık volatilite %).
    En az ~26 haftalık getiri (yaklaşık 6 ay) yoksa (nan, nan)."""
    s = pd.Series(s).dropna()
    if len(s) < 20:
        return np.nan, np.nan
    w = s.resample("W-FRI").last().dropna()
    r = w.pct_change().dropna()
    if len(r) < 26:
        return np.nan, np.nan
    vol = float(r.std() * np.sqrt(52) * 100)
    for esik, d in RISK_BANTLARI:
        if vol < esik:
            return d, vol
    return 7, vol


RISK_ACIKLAMA = (
    "TEFAS'ta her fonun yanında görünen **Risk Değeri (1-7)**, SPK düzenlemesine (Yatırımcı Bilgi Formu) göre "
    "fonun geçmiş fiyat oynaklığından hesaplanır: **haftalık getirilerin yıllık standart sapması** aşağıdaki "
    "bantlara çevrilir. Bu panel aynı bantları ve haftalık getiri yöntemini kullanır.\n\n"
    "**Fark:** TEFAS genellikle son 5 yılı kullanır; panel yan paneldeki tarih aralığını kullanır (en az ~6 ay veri "
    "gerekir). Bu yüzden panelin hesapladığı değer TEFAS'taki resmi değerden farklı çıkabilir; resmi değer için "
    "fonun TEFAS sayfasına veya Yatırımcı Bilgi Formu'na bakın. Risk değeri anapara garantisi anlamına gelmez; "
    "likidite ve kredi riskini ölçmez, geçmişe bakar.\n\n")
RISK_BANT_MD = (
    "| Risk değeri | Yıllık volatilite (haftalık getirilerden) | Tipik örnekler (yaklaşık) |\n|---|---|---|\n"
    "| **1** | %0,5'ten az | Likit ve gecelik repo fonları |\n"
    "| **2** | %0,5 – %2 | TL para piyasası fonları |\n"
    "| **3** | %2 – %5 | Kısa/orta vadeli borçlanma, bazı karma fonlar |\n"
    "| **4** | %5 – %10 | Dengeli karma, bazı altın ve döviz fonları |\n"
    "| **5** | %10 – %15 | Hisse ağırlıklı karma, bazı endeks fonları |\n"
    "| **6** | %15 – %25 | Hisse ve sektör fonları |\n"
    "| **7** | %25 ve üzeri | Yoğunlaşmış hisse, serbest ve kaldıraçlı fonlar |\n")


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
            "Risk değeri (1-7)": risk_degeri(s)[0],
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


def _risk_chip(s):
    d, _ = risk_degeri(s)
    return "" if pd.isna(d) else f"<span class='chip flat'>Risk {int(d)}/7</span>"


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
            f"<div class='chips'>{_chip('1G', d1)}{_chip('1A', m1)}{_chip('YBB', ytd)}{_risk_chip(s)}</div></div>")
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


def var_text(a):
    """VaR güven düzeyi açıklaması (yan panel ve risk bölümlerinde ortak kullanılır)."""
    n = int(round(1 / (1 - a)))
    q = int(round(a * 100))
    return (
        "**VaR (Riske Maruz Değer) nedir?**  \n"
        "Geçmiş verilere göre, bir işlem gününde belirli bir olasılıkla **aşılmayan kayıp eşiği**dir. "
        "Örneğin %1,5 VaR, 'günlük kayıp genellikle %1,5'i geçmez' demektir.\n\n"
        "**Güven düzeyi** bu 'genellikle'nin ne kadar sık olduğunu belirler:\n"
        "- **%90:** 10 günde 1 kez aşılır. Eşik küçüktür; sık yaşanan kötü günleri gösterir.\n"
        "- **%95:** 20 günde 1 kez aşılır. En yaygın kullanılan düzeydir, dengeli bir bakış verir.\n"
        "- **%99:** 100 günde 1 kez aşılır. Eşik büyüktür; nadir ama ağır kayıpları gösterir, "
        "temkinli/stres bakışı içindir.\n\n"
        f"**Şu an seçili: %{q}.** Kayıp işlem günlerinin %{q}'inde VaR'ın altında kalır; "
        f"yaklaşık {n} günde bir bu eşik aşılır ve o gün kayıp daha büyük olur.\n\n"
        f"**Örnek:** %{q} VaR %1,5 ve tutarınız 100.000 TL ise, geçmişte 100 işlem gününden {q} tanesinde "
        "günlük kayıp 1.500 TL'yi geçmemiştir.\n\n"
        "**CVaR (Beklenen Kayıp):** VaR eşiğinin aşıldığı en kötü günlerdeki **ortalama kayıp**tır. "
        "VaR'dan büyüktür ve 'kötü gün ne kadar kötü olur?' sorusunu yanıtlar.\n\n"
        "**Hangisini seçmeli?** Genel bakış için %95; daha temkinli bir stres bakışı için %99; sık "
        "dalgalanmayı görmek için %90. Bu seçim yalnızca VaR/CVaR değerlerini değiştirir, getiriyi veya "
        "Sharpe'ı etkilemez. Değerler geçmiş verilere dayanır; gelecekteki kaybın üst sınırı değildir.")


# ---------- Arayüz (sade) ----------
# Yan panel kaldırıldı; ayarlar sabit varsayılanlardır. Değiştirmek için aşağıdaki değerleri düzenleyin.
live = True                                   # canlı TEFAS verisi
rf = 0.40                                     # yıllık risksiz oran (yer tutucu, güncel orana göre değiştirin)
alpha = 0.95                                  # VaR güven düzeyi
end = pd.Timestamp.today().normalize()
start = end - pd.DateOffset(years=1)          # metinde dönem yazılmazsa son 1 yıl

# bulten_panel / profil_panel'in globals() üzerinden beklediği adlar
p = pd.DataFrame()
codes, titles_all, bench = [], {}, None
th = THEMES["Açık"]

st.markdown(make_css(th), unsafe_allow_html=True)
render_profil_form(globals())     # yalnızca yatırımcı profili oluşturma bölümü
st.divider()
render_bulten(globals())          # "ne incelemek istiyorsun" kutusu + 4 işaret + sepet akışı yazısı

st.divider()
st.caption("Bu panel bilgilendirme amaçlıdır, yatırım tavsiyesi değildir. "
           "Geçmiş performans gelecekteki getirinin garantisi değildir. "
           "Veriler TEFAS ve Yahoo Finance kaynaklıdır; kritik rakamları resmi kaynaktan teyit ediniz.")
