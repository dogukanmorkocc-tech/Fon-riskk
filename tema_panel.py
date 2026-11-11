"""tema_panel.py - Sayfanın genel görünümü (CSS) ve yasal uyarı metinleri.

Renkler: sıcak krem zemin, kiremit vurgu, yeşil/kırmızı yalnızca artı/eksi için.
Grafik renkleri app.py'deki TEMA sözlüğünden gelir (aşağıdaki TEMA ile aynıdır).
"""
import streamlit as st

TEMA_RENKLERI = dict(
    bg="#faf9f5", card="#ffffff", card2="#f4efe6", border="#e8e2d6", text="#1f1e1d", muted="#6f6a60",
    accent="#c96442", up="#2f7d4f", down="#b3402a", upbg="#e3f1e8", downbg="#f8e3de", flatbg="#f1ede4",
    palette=["#c96442", "#2f6f6d", "#d4a24c", "#5b6abf", "#a34a6b", "#3f8f5f", "#6f6a60", "#b5543a"],
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea { font-family: 'Inter','Segoe UI',-apple-system,sans-serif; }
.stApp { background: #faf9f5; color: #1f1e1d; }
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { font-family: 'Source Serif 4', Georgia, serif !important; color: #1f1e1d !important; letter-spacing: -.2px; }
h3 { font-size: 1.55rem !important; margin-top: .4rem !important; }
h4 { font-size: 1.2rem !important; }
h5 { font-size: .98rem !important; font-weight: 600 !important; color: #3b3832 !important; margin-top: .8rem !important; }
hr { border-color: #e8e2d6 !important; }

/* sekmeler */
div[data-baseweb="tab-list"] { gap: 6px; background: #f1ede4; padding: 5px; border-radius: 14px; width: fit-content; max-width: 100%; }
button[data-baseweb="tab"] { height: 42px; border-radius: 10px; padding: 0 20px; font-weight: 600; color: #6f6a60; background: transparent; }
button[data-baseweb="tab"][aria-selected="true"] { background: #ffffff; color: #c96442; box-shadow: 0 1px 4px rgba(60,40,20,.14); }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

/* girdiler */
div[data-testid="stTextInput"] input { border-radius: 14px; border: 1.5px solid #e0d9ca; background: #fff; padding: 14px 16px; font-size: 16px; }
div[data-testid="stTextInput"] input:focus { border-color: #c96442; box-shadow: 0 0 0 3px rgba(201,100,66,.15); }
div[data-testid="stCheckbox"] label { background: #fff; border: 1px solid #e8e2d6; border-radius: 12px; padding: 8px 12px; width: 100%; }
div[data-testid="stCheckbox"] label:hover { border-color: #c96442; }
.stButton > button, .stFormSubmitButton > button { border-radius: 12px; font-weight: 600; padding: 10px 22px; border: 1px solid #e0d9ca; background: #fff; color: #1f1e1d; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stFormSubmitButton > button {
  background: #c96442; border-color: #c96442; color: #fff; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover { background: #b0532f; border-color: #b0532f; color: #fff; }
div[data-testid="stForm"] { background: #fff; border: 1px solid #e8e2d6; border-radius: 16px; padding: 18px 20px; }
div[data-testid="stExpander"] { background: #fff; border: 1px solid #e8e2d6; border-radius: 14px; }
div[data-testid="stAlert"] { border-radius: 14px; }
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] { border: 1px solid #e8e2d6; border-radius: 14px; overflow: hidden; }
div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] { background: #fff; border: 1px solid #e8e2d6; border-radius: 14px; padding: 8px; }

/* yasal uyarı */
.legal { background: #fff7f3; border: 2px solid #c96442; border-radius: 16px; padding: 18px 22px; margin: 4px 0 18px; }
.legal .lt { font-family: 'Source Serif 4', Georgia, serif; font-size: 22px; font-weight: 700; color: #8a3b25; margin-bottom: 8px; }
.legal .lb { font-size: 17px; line-height: 1.55; color: #2b1d17; font-weight: 600; }
.legal ul { margin: 10px 0 6px 18px; padding: 0; }
.legal li { font-size: 14.5px; line-height: 1.5; color: #3b2a22; margin-bottom: 5px; }
.legal .ln { font-size: 12.5px; color: #6f5a50; margin-top: 8px; }
.legal-mini { font-size: 12.5px; color: #6f6a60; border-top: 1px solid #e8e2d6; padding-top: 12px; margin-top: 28px; line-height: 1.5; }

/* sepet bölüm başlığı */
.bolum { display:flex; align-items:center; gap:10px; margin: 8px 0 2px; }
.bolum .nk { width: 34px; height: 34px; border-radius: 10px; background: #c96442; color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; }
.surum { font-size: 11px; color: #a39d90; margin-top: 6px; }
@media (max-width: 640px) {
  .block-container { padding-left: .9rem; padding-right: .9rem; }
  .legal { padding: 14px 15px; } .legal .lb { font-size: 15.5px; } .legal .lt { font-size: 19px; }
  button[data-baseweb="tab"] { padding: 0 12px; }
}
</style>
"""

YASAL_BASLIK = "⚠ ÖNEMLİ YASAL UYARI – YATIRIM DANIŞMANLIĞI DEĞİLDİR"
YASAL_ANA = ("Bu sayfada yer alan hiçbir içerik (fon sepeti, grafik, tablo, yazı veya hesaplama) yatırım danışmanlığı "
             "değildir ve bu amaçla sunulmamaktadır. Burada yatırım danışmanlığı hizmeti verilmemektedir.")
YASAL_MADDELER = [
    "Yatırım danışmanlığı; 6362 sayılı Sermaye Piyasası Kanunu ve ilgili SPK düzenlemeleri uyarınca yalnızca "
    "yetkilendirilmiş kuruluşlarca, kişinin mali durumu ile risk ve getiri tercihleri dikkate alınarak verilebilir. "
    "Bu sayfa böyle bir yetkiye sahip bir kuruluş adına hizmet sunmamaktadır.",
    "Sunulan sepetler, geçmiş verilere dayalı <b>örnek simülasyonlardır</b>; herhangi bir fon, ETF veya "
    "menkul kıymetin alınması, satılması ya da tutulması yönünde tavsiye, teklif veya yönlendirme niteliği taşımaz. "
    "“Yatırımcı profili” anketi, SPK mevzuatındaki uygunluk/yerindelik testinin yerine geçmez.",
    "Geçmiş performans, gelecekteki getirinin garantisi veya göstergesi değildir. Yatırım fonları ve ETF'ler "
    "anapara kaybı dahil zarar riski taşır.",
    "Veriler TEFAS ve Yahoo Finance gibi üçüncü kaynaklardan alınır; gecikmeli, eksik veya hatalı olabilir, doğruluğu "
    "ve güncelliği garanti edilmez. Stopaj, vergi ve masraf hesapları tahminîdir.",
    "Karar vermeden önce fon izahnamesini, KAP duyurularını ve resmi kaynakları inceleyin; gerekirse yetkili bir "
    "yatırım kuruluşundan veya lisanslı bir danışmandan destek alın.",
    "Bu içeriğe dayanılarak alınan kararlardan doğabilecek kâr ve zararlar tamamen kullanıcıya aittir; sayfa sahibi, "
    "kanunun emredici hükümleri saklı kalmak üzere, bu nedenle doğabilecek zararlardan sorumlu tutulamaz.",
]
YASAL_ALT = "Bu sayfayı kullanarak yukarıdaki koşulları okuduğunuzu ve kabul ettiğinizi beyan etmiş sayılırsınız."
ONAY_METNI = "Yasal uyarıyı okudum; bu sayfadaki içeriğin yatırım danışmanlığı olmadığını ve tavsiye niteliği taşımadığını anladım."


def css():
    st.markdown(CSS, unsafe_allow_html=True)


def yasal_uyari():
    li = "".join(f"<li>{m}</li>" for m in YASAL_MADDELER)
    st.markdown(f"<div class='legal'><div class='lt'>{YASAL_BASLIK}</div><div class='lb'>{YASAL_ANA}</div>"
                f"<ul>{li}</ul><div class='ln'>{YASAL_ALT}</div></div>", unsafe_allow_html=True)


def yasal_alt():
    st.markdown("<div class='legal-mini'><b>Yasal uyarı:</b> Bu sayfada yatırım danışmanlığı verilmemektedir. "
                "İçerik yalnızca genel bilgilendirme amaçlı örnek simülasyondur; tavsiye, teklif veya yönlendirme değildir. "
                "Geçmiş performans gelecekteki getiriyi garanti etmez; anapara kaybı dahil zarar riski vardır. "
                "Kararlarınızın sorumluluğu size aittir.</div>", unsafe_allow_html=True)
