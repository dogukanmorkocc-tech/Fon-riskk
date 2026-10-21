"""TEFAS Fon Performans & Risk Prototipi (Streamlit)
Çalıştır: streamlit run app.py
Veri: tefas-crawler (canlı) veya sentetik demo verisi.
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fon Risk Analizi", layout="wide")
TRADING_DAYS = 252


# ---------- Veri ----------
@st.cache_data(show_spinner="Veri çekiliyor...", ttl=3600)
def load_tefas(codes, start, end):
    from tefas import Crawler  # pip install tefas-crawler
    crawler = Crawler()
    frames = []
    for code in codes:
        df = crawler.fetch(start=start, end=end, name=code,
                           columns=["code", "date", "price"])
        frames.append(df)
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="code", values="price").sort_index()


@st.cache_data
def load_synthetic(codes, start, end, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    data = {}
    for c in codes:
        mu, sigma = rng.uniform(0.0003, 0.0012), rng.uniform(0.002, 0.012)
        r = rng.normal(mu, sigma, len(idx))
        r[rng.integers(0, len(idx), 2)] -= 4 * sigma  # anomali tohumu
        data[c] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


# ---------- Metrikler ----------
def risk_table(prices, bench=None, rf=0.40, alpha=0.95):
    rets = prices.pct_change().dropna()
    rows = {}
    for c in rets:
        r = rets[c]
        ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        vol = r.std() * np.sqrt(TRADING_DAYS)
        dd = (prices[c] / prices[c].cummax() - 1).min()
        var = -np.percentile(r, (1 - alpha) * 100)
        cvar = -r[r <= -var].mean()
        row = {
            "Yıllık getiri": ann_ret,
            "Yıllık volatilite": vol,
            "Sharpe": (ann_ret - rf) / vol if vol else np.nan,
            "Maks. düşüş": dd,
            f"VaR {int(alpha*100)}% (1g)": var,
            f"CVaR {int(alpha*100)}% (1g)": cvar,
        }
        if bench is not None and c != bench:
            te = (r - rets[bench]).std() * np.sqrt(TRADING_DAYS)
            row["Tracking error"] = te
        rows[c] = row
    return pd.DataFrame(rows).T


def detect_anomalies(prices, window=60, z=3.0):
    rets = prices.pct_change()
    zs = (rets - rets.rolling(window).mean()) / rets.rolling(window).std()
    out = zs.stack().rename("z").reset_index()
    out.columns = ["Tarih", "Fon", "z"]
    out["Getiri"] = [rets.loc[d, f] for d, f in zip(out["Tarih"], out["Fon"])]
    return out[out["z"].abs() >= z].sort_values("Tarih", ascending=False)


# ---------- Arayüz ----------
st.title("TEFAS Fon Performans & Risk Paneli")

with st.sidebar:
    codes = [c.strip().upper() for c in
             st.text_input("Fon kodları (virgülle)", "AFA,TI2,IPB,YAS").split(",") if c.strip()]
    start = st.date_input("Başlangıç", pd.Timestamp.today() - pd.DateOffset(years=1))
    end = st.date_input("Bitiş", pd.Timestamp.today())
    live = st.toggle("Canlı TEFAS verisi", value=False,
                     help="Kapalıyken sentetik demo verisi kullanılır.")
    rf = st.number_input("Risksiz oran (yıllık)", 0.0, 1.0, 0.40, 0.01)
    alpha = st.select_slider("VaR güven düzeyi", [0.90, 0.95, 0.99], 0.95)

if not codes:
    st.stop()

try:
    prices = (load_tefas(tuple(codes), str(start), str(end)) if live
              else load_synthetic(tuple(codes), str(start), str(end)))
except Exception as e:
    st.error(f"Veri alınamadı: {e}")
    st.stop()

bench = st.sidebar.selectbox("Kıyas fonu (tracking error)", prices.columns)

tab1, tab2, tab3 = st.tabs(["Performans", "Risk metrikleri", "Anomaliler"])
with tab1:
    st.subheader("Normalize fiyat (başlangıç = 100)")
    st.line_chart(prices / prices.iloc[0] * 100)
with tab2:
    tbl = risk_table(prices, bench, rf, alpha)
    st.dataframe(tbl.style.format("{:.2%}", subset=[c for c in tbl if c != "Sharpe"])
                 .format("{:.2f}", subset=["Sharpe"]), use_container_width=True)
    st.subheader("Getiri korelasyonu")
    st.dataframe(prices.pct_change().corr().style.background_gradient(cmap="RdYlGn"),
                 use_container_width=True)
with tab3:
    st.caption("60 günlük kayan pencerede |z| ≥ 3 olan günlük getiriler")
    st.dataframe(detect_anomalies(prices), use_container_width=True)
