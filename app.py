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
    return f"%{x * 100:.2f}".replace(".", ",")


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


# ---------- Arayüz ----------
with st.sidebar:
    st.header("Ayarlar")
    title = st.text_input("Panel başlığı", "Fon Analiz Paneli")
    codes = [c.strip().upper() for c in
             st.text_input("Fon kodları (virgülle)", "AFA,TI2,IPB,YAS").split(",")
             if c.strip()]
    start = st.date_input("Başlangıç", pd.Timestamp.today() - pd.DateOffset(years=1))
    end = st.date_input("Bitiş", pd.Timestamp.today())
    live = st.toggle("Canlı TEFAS verisi", value=True,
                     help="Kapalıyken sentetik demo verisi kullanılır.")
    rf = st.number_input("Risksiz oran (yıllık)", 0.0, 1.0, 0.40, 0.01)
    alpha = st.select_slider("VaR güven düzeyi", [0.90, 0.95, 0.99], 0.95)

st.title(title)
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

tabs = st.tabs(["Özet", "Performans", "Risk", "Fon büyüklüğü ve akış",
                "Anomaliler", "Metodoloji ve tutarlılık", "Yatırımcı karşılaştırma"])

with tabs[0]:
    st.subheader("Dönemsel getiri ve sıralama")
    pr = period_returns(p)
    st.dataframe(pr.style.format({c: "{:.2%}" for c in pr.columns if c != "Sıra (3 Ay)"}),
                 use_container_width=True)

with tabs[1]:
    st.subheader("Normalize fiyat (başlangıç = 100)")
    base = p.apply(lambda s: s.dropna().iloc[0])
    st.line_chart(pf / base * 100)
    st.subheader("Zirveden düşüş (drawdown)")
    st.line_chart(pf / pf.cummax() - 1)

with tabs[2]:
    rt = risk_table(p, bench, rf, alpha)
    fm = {c: "{:.2%}" for c in rt.columns if c != "Sharpe"}
    fm["Sharpe"] = "{:.2f}"
    st.dataframe(rt.style.format(fm), use_container_width=True)
    if (p.index[-1] - p.index[0]).days < 90:
        st.warning("Dönem 90 günden kısa: yıllıklandırılmış değerler yanıltıcı olabilir.")
    st.download_button("Tabloyu indir (CSV)", rt.to_csv().encode("utf-8-sig"),
                       "fon_risk.csv", "text/csv")
    st.subheader("Risk-getiri dağılımı")
    sc = rt.reset_index().rename(columns={"index": "Fon"})
    st.scatter_chart(sc, x="Yıllık volatilite", y="Yıllık getiri", color="Fon")
    st.subheader("30 gözlemlik kayan volatilite (yıllıklandırılmış)")
    R = returns_df(p)
    rv = R.rolling(30, min_periods=20).std()
    st.line_chart(rv * np.sqrt(pd.Series({c: obs_per_year(p[c]) for c in p})))
    st.subheader("Getiri korelasyonu (ortak tarihler)")
    st.dataframe(R.corr().style.format("{:.2f}"), use_container_width=True)

with tabs[3]:
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
        st.line_chart(cum)

with tabs[4]:
    st.caption("60 gözlemlik kayan pencerede |z| >= 3 olan günlük getiriler")
    st.dataframe(detect_anomalies(p), use_container_width=True)

with tabs[5]:
    st.subheader("Hesaplama yöntemi")
    st.markdown(
        "- **Basit getiri:** r_t = P_t / P_(t-1) - 1 (her fon kendi gözlem tarihlerinde; "
        "veri boşlukları doldurulmaz, boşluk üzerindeki getiri tek bir çok günlük getiridir)\n"
        "- **Dönem getirisi:** R = P_T / P_0 - 1 = Π(1 + r_t) - 1 (zincirleme bağlama)\n"
        "- **Yıllıklandırma:** (1 + R)^(365,25 / takvim günü) - 1\n"
        "- **Yıllık gözlem sayısı:** n / (takvim günü / 365,25); sabit 252 varsayılmaz\n"
        "- **Volatilite:** std(r) x √(yıllık gözlem)\n"
        "- **Sharpe:** ortalama(r - rf_g) / std(r - rf_g) x √(yıllık gözlem); "
        "rf_g = (1 + rf)^(1 / yıllık gözlem) - 1\n"
        "- **VaR / CVaR:** tarihsel simülasyon, 1 gözlemlik getiri\n"
        "- **Tracking error:** ortak tarihlerde (r_fon - r_kıyas) std x √(yıllık gözlem)\n"
        "- **Net akış:** (pay_t - pay_(t-1)) x fiyat_t; pay sayısı yoksa fon büyüklüğü / fiyat")
    st.subheader("Tutarlılık kontrolü")
    st.caption("Fiyattan, zincirleme ve log toplamından hesaplanan dönem getirileri "
               "aynı olmalı. Fark 1e-9'dan büyükse veri veya hesap sorunu vardır.")
    cc = consistency_checks(p)
    pct = ["Fiyattan dönem getirisi", "Zincirleme getiri", "Log toplamından getiri",
           "Yıllık (takvim günü)", "Yıllık (252 gün varsayımı)", "En büyük günlük hareket"]
    st.dataframe(cc.style.format({**{c: "{:.4%}" for c in pct}, "Fark": "{:.1e}"}),
                 use_container_width=True)
    st.caption("'Yıllık (252 gün varsayımı)' ile takvim günü bazlı değer arasındaki "
               "fark, sabit gün sayısı varsaymanın etkisini gösterir. "
               "Eksik iş günlerinin çoğu resmi tatillerdir.")

with tabs[6]:
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
