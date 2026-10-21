"""TEFAS Fon Analiz Paneli v2 - banka hazine / fon ürün yönetimi prototipi
Çalıştır: streamlit run app.py
Ek kütüphane gerekmez (requirements.txt aynı kalabilir).
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
            w = df.pivot_table(index="date", columns="code", values=col,
                               aggfunc="last").sort_index()
            out[col] = w
    out["price"] = out["price"].ffill().dropna(how="all")
    # Pay sayısı gelmezse: pay sayısı = fon büyüklüğü / fiyat
    if "number_of_shares" not in out and "market_cap" in out:
        out["number_of_shares"] = (out["market_cap"].ffill() / out["price"]) \
            .replace([np.inf, -np.inf], np.nan)
    return out


# ---------- Metrikler ----------
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
    rets = p.pct_change().dropna(how="all")
    rows = {}
    for c in rets:
        r = rets[c].dropna()
        if len(r) < 20:
            continue
        ann = (1 + r).prod() ** (TD / len(r)) - 1
        vol = r.std() * np.sqrt(TD)
        dd = (p[c] / p[c].cummax() - 1).min()
        var = -np.percentile(r, (1 - alpha) * 100)
        cvar = -r[r <= -var].mean()
        row = {"Yıllık getiri": ann, "Yıllık volatilite": vol,
               "Sharpe": (ann - rf) / vol if vol else np.nan,
               "Maks. düşüş": dd,
               f"VaR %{int(alpha*100)} (1g)": var,
               f"CVaR %{int(alpha*100)} (1g)": cvar}
        if c != bench and bench in rets:
            row["Tracking error"] = (r - rets[bench]).dropna().std() * np.sqrt(TD)
        rows[c] = row
    return pd.DataFrame(rows).T


def flow_table(w):
    """Fon büyüklüğü, yatırımcı sayısı ve tahmini net akış (Δpay x fiyat)."""
    if "number_of_shares" not in w:
        return None, None
    sh = w["number_of_shares"].ffill()
    pr = w["price"]
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
    rets = p.pct_change()
    zs = (rets - rets.rolling(window).mean()) / rets.rolling(window).std()
    out = zs.stack().rename("z").reset_index()
    out.columns = ["Tarih", "Fon", "z"]
    out["Getiri"] = [rets.loc[d, f] for d, f in zip(out["Tarih"], out["Fon"])]
    return out[out["z"].abs() >= z].sort_values("Tarih", ascending=False)


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

tabs = st.tabs(["Özet", "Performans", "Risk", "Fon büyüklüğü ve akış", "Anomaliler"])

with tabs[0]:
    st.subheader("Dönemsel getiri ve sıralama")
    pr = period_returns(p)
    st.dataframe(pr.style.format({c: "{:.2%}" for c in pr.columns if c != "Sıra (3 Ay)"}),
                 use_container_width=True)

with tabs[1]:
    st.subheader("Normalize fiyat (başlangıç = 100)")
    st.line_chart(p / p.bfill().iloc[0] * 100)
    st.subheader("Zirveden düşüş (drawdown)")
    st.line_chart(p / p.cummax() - 1)

with tabs[2]:
    rt = risk_table(p, bench, rf, alpha)
    fm = {c: "{:.2%}" for c in rt.columns if c != "Sharpe"}
    fm["Sharpe"] = "{:.2f}"
    st.dataframe(rt.style.format(fm), use_container_width=True)
    st.download_button("Tabloyu indir (CSV)", rt.to_csv().encode("utf-8-sig"),
                       "fon_risk.csv", "text/csv")
    st.subheader("Risk-getiri dağılımı")
    sc = rt.reset_index().rename(columns={"index": "Fon"})
    st.scatter_chart(sc, x="Yıllık volatilite", y="Yıllık getiri", color="Fon")
    st.subheader("30 günlük kayan volatilite (yıllık)")
    st.line_chart(p.pct_change().rolling(30).std() * np.sqrt(TD))
    st.subheader("Getiri korelasyonu")
    st.dataframe(p.pct_change().corr().style.format("{:.2f}"), use_container_width=True)

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
        st.caption("Net akış = pay sayısı değişimi x fiyat (tahmini; fiyat etkisinden arındırılmıştır).")
        st.subheader("Kümülatif tahmini net akış (TL)")
        st.line_chart(cum)

with tabs[4]:
    st.caption("60 günlük kayan pencerede |z| >= 3 olan günlük getiriler")
    st.dataframe(detect_anomalies(p), use_container_width=True)

st.divider()
st.caption("Bu panel bilgilendirme amaçlıdır, yatırım tavsiyesi değildir. "
           "Geçmiş performans gelecekteki getirinin garantisi değildir. "
           "Veriler TEFAS kaynaklıdır; kritik rakamları resmi kaynaktan teyit ediniz.")
