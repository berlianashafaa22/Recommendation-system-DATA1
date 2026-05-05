
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib_venn import venn2
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Retail Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("Online_Retail.xlsx")
    return df

@st.cache_data
def preprocess(df):
    df_clean = df.copy()
    df_clean = df_clean[df_clean["Description"].notna()]
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean["Description"] = df_clean["Description"].str.strip().str.upper()
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["Month"] = df_clean["InvoiceDate"].dt.to_period("M").astype(str)
    df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()
    df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour
    top_products = df_clean["Description"].value_counts().head(300).index
    df_clean = df_clean[df_clean["Description"].isin(top_products)]
    return df_clean

@st.cache_data
def build_rules(df_clean):
    basket = df_clean.groupby("InvoiceNo")["Description"].apply(list).reset_index()
    basket.columns = ["InvoiceNo", "Items"]
    te = TransactionEncoder()
    te_array = te.fit_transform(basket["Items"])
    basket_encoded = pd.DataFrame(te_array, columns=te.columns_)
    frequent_itemsets = apriori(basket_encoded, min_support=0.02, use_colnames=True, max_len=2)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules

@st.cache_data
def build_cb(df_clean):
    products = df_clean[["StockCode","Description"]].drop_duplicates("Description").reset_index(drop=True)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products["Description"])
    cb_sim = cosine_similarity(tfidf_matrix)
    cb_sim_df = pd.DataFrame(cb_sim, index=products["Description"], columns=products["Description"])
    return cb_sim_df

# Fungsi rekomendasi
def apriori_only(product_name, rules, top_n=5):
    product_name = product_name.strip().upper()
    scores = {}
    filtered = rules[rules["antecedents"].apply(lambda x: product_name in x)]
    for _, row in filtered.iterrows():
        for item in row["consequents"]:
            score = row["confidence"] * row["lift"]
            if item != product_name:
                scores[item] = max(scores.get(item, 0), score)
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(result, columns=["Produk Rekomendasi", "Score"])

def cb_only(product_name, cb_sim_df, top_n=5):
    product_name = product_name.strip().upper()
    if product_name not in cb_sim_df.index:
        return pd.DataFrame(columns=["Produk Rekomendasi", "Score"])
    sim_scores = cb_sim_df[product_name].drop(product_name)
    result = sim_scores.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"Produk Rekomendasi": result.index, "Score": result.values})

def hybrid_recommend(product_name, rules, cb_sim_df, top_n=5, w_apr=0.6, w_cb=0.4):
    product_name = product_name.strip().upper()
    apr_scores = {}
    filtered = rules[rules["antecedents"].apply(lambda x: product_name in x)]
    for _, row in filtered.iterrows():
        for item in row["consequents"]:
            s = row["confidence"] * row["lift"]
            if item != product_name:
                apr_scores[item] = max(apr_scores.get(item, 0), s)
    if apr_scores:
        mx = max(apr_scores.values())
        apr_scores = {k: v/mx for k, v in apr_scores.items()}
    cb_scores = {}
    if product_name in cb_sim_df.index:
        sim = cb_sim_df[product_name].drop(product_name)
        cb_scores = sim.to_dict()
    all_items = set(list(apr_scores.keys()) + list(cb_scores.keys()))
    hybrid = {item: w_apr * apr_scores.get(item, 0) + w_cb * cb_scores.get(item, 0) for item in all_items}
    result = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(result, columns=["Produk Rekomendasi", "Score"])

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Memuat dan memproses data..."):
    df_raw = load_data()
    df_clean = preprocess(df_raw)
    rules = build_rules(df_clean)
    cb_sim_df = build_cb(df_clean)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
    st.title("🛒 RetailRec")
    st.caption("Recommendation System Dashboard")
    st.divider()

    mode = st.radio(
        "Pilih Mode Tampilan",
        ["👔 Executive Dashboard", "🔬 Analyst View"],
        index=0
    )
    st.divider()

    page = st.selectbox(
        "Navigasi Halaman",
        ["🏠 Home",
         "📊 Eksplorasi Data",
         "🤖 Sistem Rekomendasi",
         "🔬 Analisis Rules",
         "📈 Evaluasi Metode",
         "💼 Implikasi Manajerial"]
    )
    st.divider()
    st.caption(f"📦 Total data: {len(df_raw):,} baris")
    st.caption(f"🌍 {df_raw['Country'].nunique()} negara")
    st.caption(f"📅 {df_raw['InvoiceDate'].min().date()} s/d {df_raw['InvoiceDate'].max().date()}")

# ─────────────────────────────────────────
# HALAMAN 1: HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("🛒 Retail Recommendation System")
    st.subheader("Global Market Basket Analysis & Hybrid Recommendation")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transaksi", f"{len(df_raw):,}")
    col2.metric("Produk Unik", f"{df_clean['Description'].nunique():,}")
    col3.metric("Negara", f"{df_raw['Country'].nunique()}")
    col4.metric("Association Rules", f"{len(rules):,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **📌 Insight Utama**
        - Ditemukan **{len(rules)} pola pembelian** yang kuat secara global
        - Rata-rata lift **{rules['lift'].mean():.1f}x** — produk yang direkomendasikan jauh lebih sering dibeli bersama dibanding kebetulan
        - Confidence tertinggi **{rules['confidence'].max():.0%}** — ada pasangan produk yang hampir selalu dibeli bersamaan
        """)
    with col2:
        st.success(f"""
        **🎯 Tentang Sistem Ini**
        - Menggunakan pendekatan **Hybrid Recommendation**
        - Menggabungkan **Apriori** (pola pembelian) + **Content-Based** (kemiripan produk)
        - Coverage rekomendasi: **100% produk** dapat direkomendasikan
        - Data mencakup **{df_raw["Country"].nunique()} negara** secara global
        """)

    st.markdown("---")
    st.markdown("### 🚀 Mulai Eksplorasi")
    col1, col2, col3 = st.columns(3)
    col1.info("📊 **Eksplorasi Data**
Lihat distribusi dan pola dalam data")
    col2.success("🤖 **Sistem Rekomendasi**
Coba rekomendasi produk secara interaktif")
    col3.warning("💼 **Implikasi Manajerial**
Temuan dan rekomendasi aksi bisnis")

# ─────────────────────────────────────────
# HALAMAN 2: EDA
# ─────────────────────────────────────────
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📦 Distribusi Data", "📅 Analisis Temporal", "🔍 Kualitas Data"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 Produk Terlaris")
            top_prod = df_clean.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                top_prod.sort_values(),
                orientation="h",
                labels={"value": "Total Quantity", "index": "Produk"},
                color=top_prod.sort_values().values,
                color_continuous_scale="Blues"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 Negara")
            top_country = df_raw["Country"].value_counts().head(10)
            fig = px.bar(
                top_country.sort_values(),
                orientation="h",
                labels={"value": "Jumlah Transaksi", "index": "Negara"},
                color=top_country.sort_values().values,
                color_continuous_scale="Oranges"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribusi Quantity & UnitPrice")
        col1, col2 = st.columns(2)
        df_pos = df_raw[(df_raw["Quantity"] > 0) & (df_raw["UnitPrice"] > 0)]
        with col1:
            fig = px.histogram(
                df_pos[df_pos["Quantity"] <= df_pos["Quantity"].quantile(0.95)],
                x="Quantity", nbins=50, title="Distribusi Quantity",
                color_discrete_sequence=["steelblue"]
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(
                df_pos[df_pos["UnitPrice"] <= df_pos["UnitPrice"].quantile(0.95)],
                x="UnitPrice", nbins=50, title="Distribusi UnitPrice",
                color_discrete_sequence=["mediumseagreen"]
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Tren Transaksi per Bulan")
        monthly = df_clean.groupby("Month")["InvoiceNo"].nunique().reset_index()
        monthly.columns = ["Bulan", "Jumlah Invoice"]
        fig = px.line(monthly, x="Bulan", y="Jumlah Invoice", markers=True,
                      color_discrete_sequence=["steelblue"])
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        peak_month = monthly.loc[monthly["Jumlah Invoice"].idxmax(), "Bulan"]
        st.info(f"📌 **Insight:** Transaksi tertinggi terjadi pada **{peak_month}**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transaksi per Hari")
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily = df_clean["DayOfWeek"].value_counts().reindex(day_order).reset_index()
            daily.columns = ["Hari", "Jumlah"]
            fig = px.bar(daily, x="Hari", y="Jumlah", color_discrete_sequence=["mediumseagreen"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Transaksi per Jam")
            hourly = df_clean["Hour"].value_counts().sort_index().reset_index()
            hourly.columns = ["Jam", "Jumlah"]
            fig = px.bar(hourly, x="Jam", y="Jumlah", color_discrete_sequence=["coral"])
            st.plotly_chart(fig, use_container_width=True)

        peak_day = daily.loc[daily["Jumlah"].idxmax(), "Hari"]
        peak_hour = hourly.loc[hourly["Jumlah"].idxmax(), "Jam"]
        st.info(f"📌 **Insight:** Hari tersibuk adalah **{peak_day}** dan jam tersibuk adalah **{peak_hour}:00**")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Missing Value")
            missing = df_raw.isnull().sum().reset_index()
            missing.columns = ["Kolom", "Jumlah Missing"]
            missing["Persentase (%)"] = (missing["Jumlah Missing"] / len(df_raw) * 100).round(2)
            st.dataframe(missing, use_container_width=True)

        with col2:
            st.subheader("Transaksi Cancel vs Valid")
            cancel = df_raw[df_raw["InvoiceNo"].astype(str).str.startswith("C")]
            non_cancel = df_raw[~df_raw["InvoiceNo"].astype(str).str.startswith("C")]
            fig = px.pie(
                values=[len(non_cancel), len(cancel)],
                names=["Valid", "Cancel"],
                color_discrete_sequence=["#66b3ff", "#ff9999"]
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ringkasan Preprocessing")
        summary = pd.DataFrame({
            "Tahap": [
                "Data Awal",
                "Setelah hapus missing Description",
                "Setelah hapus transaksi cancel",
                "Setelah hapus nilai negatif",
                "Setelah filter top 300 produk"
            ],
            "Jumlah Baris": [
                len(df_raw),
                len(df_raw[df_raw["Description"].notna()]),
                len(df_raw[df_raw["Description"].notna() & ~df_raw["InvoiceNo"].astype(str).str.startswith("C")]),
                len(df_raw[df_raw["Description"].notna() & ~df_raw["InvoiceNo"].astype(str).str.startswith("C") & (df_raw["Quantity"] > 0) & (df_raw["UnitPrice"] > 0)]),
                len(df_clean)
            ]
        })
        summary["Perubahan"] = summary["Jumlah Baris"].diff().fillna(0).astype(int)
        st.dataframe(summary, use_container_width=True)

# ─────────────────────────────────────────
# HALAMAN 3: SISTEM REKOMENDASI
# ─────────────────────────────────────────
elif page == "🤖 Sistem Rekomendasi":
    st.title("🤖 Sistem Rekomendasi Produk")
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        product_list = sorted(df_clean["Description"].unique().tolist())
        selected_product = st.selectbox("🔍 Pilih Produk", product_list)

    with col2:
        top_n = st.slider("Jumlah Rekomendasi", min_value=3, max_value=10, value=5)

    with col3:
        method = st.radio(
            "Metode",
            ["Hybrid", "Apriori", "Content-Based"],
            index=0
        )

    st.markdown("---")

    if st.button("🚀 Cari Rekomendasi", use_container_width=True):
        if method == "Hybrid":
            result = hybrid_recommend(selected_product, rules, cb_sim_df, top_n)
            method_desc = "Hybrid (60% Apriori + 40% Content-Based)"
            color = "🟡"
        elif method == "Apriori":
            result = apriori_only(selected_product, rules, top_n)
            method_desc = "Apriori (Pola Pembelian)"
            color = "🔵"
        else:
            result = cb_only(selected_product, cb_sim_df, top_n)
            method_desc = "Content-Based (Kemiripan Produk)"
            color = "🟢"

        st.subheader(f"{color} Rekomendasi untuk: **{selected_product}**")
        st.caption(f"Metode: {method_desc}")

        if len(result) == 0:
            st.warning("⚠️ Tidak ada rekomendasi ditemukan untuk produk ini dengan metode yang dipilih. Coba gunakan metode Hybrid atau Content-Based.")
        else:
            for i, row in result.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    col1.markdown(f"**{i+1}. {row['Produk Rekomendasi']}**")
                    col2.progress(float(min(row["Score"], 1.0)))
            st.markdown("---")

        # Bandingkan semua metode
        st.subheader("📊 Perbandingan Semua Metode")
        res_apr = apriori_only(selected_product, rules, top_n)
        res_cb = cb_only(selected_product, cb_sim_df, top_n)
        res_hybrid = hybrid_recommend(selected_product, rules, cb_sim_df, top_n)

        def pad(lst, n):
            return lst + ["-"] * (n - len(lst))

        comparison = pd.DataFrame({
            "Rank": range(1, top_n + 1),
            "🔵 Apriori": pad(res_apr["Produk Rekomendasi"].tolist(), top_n),
            "🟢 Content-Based": pad(res_cb["Produk Rekomendasi"].tolist(), top_n),
            "🟡 Hybrid": pad(res_hybrid["Produk Rekomendasi"].tolist(), top_n)
        }).set_index("Rank")

        st.dataframe(comparison, use_container_width=True)

# ─────────────────────────────────────────
# HALAMAN 4: ANALISIS RULES
# ─────────────────────────────────────────
elif page == "🔬 Analisis Rules":
    st.title("🔬 Analisis Association Rules")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    min_support = col1.slider("Min Support", 0.01, 0.1, 0.02, 0.01)
    min_confidence = col2.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    min_lift = col3.slider("Min Lift", 1.0, 15.0, 1.0, 0.5)

    filtered_rules = rules[
        (rules["support"] >= min_support) &
        (rules["confidence"] >= min_confidence) &
        (rules["lift"] >= min_lift)
    ].copy()

    filtered_rules["antecedents"] = filtered_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    filtered_rules["consequents"] = filtered_rules["consequents"].apply(lambda x: ", ".join(list(x)))

    st.metric("Rules yang memenuhi filter", len(filtered_rules))
    st.dataframe(
        filtered_rules[["antecedents","consequents","support","confidence","lift"]]
        .round(4)
        .reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📈 Scatter Plot Metrik")

    fig = px.scatter(
        filtered_rules,
        x="support", y="confidence",
        color="lift", size="lift",
        hover_data=["antecedents","consequents","lift"],
        color_continuous_scale="YlOrRd",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        title="Support vs Confidence (ukuran & warna = Lift)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top 10 Rules Terkuat")
    top10 = rules.head(10).copy()
    top10["antecedents"] = top10["antecedents"].apply(lambda x: ", ".join(list(x)))
    top10["consequents"] = top10["consequents"].apply(lambda x: ", ".join(list(x)))

    for i, row in top10.iterrows():
        with st.expander(f"#{i+1} — {row['antecedents']} → {row['consequents']} | Lift: {row['lift']:.2f}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Support", f"{row['support']:.4f}")
            col2.metric("Confidence", f"{row['confidence']:.4f}")
            col3.metric("Lift", f"{row['lift']:.4f}")
            st.caption(f"Artinya: {row['confidence']*100:.1f}% pembeli **{row['antecedents']}** juga membeli **{row['consequents']}**, dan hubungan ini {row['lift']:.1f}x lebih kuat dari kebetulan.")

# ─────────────────────────────────────────
# HALAMAN 5: EVALUASI
# ─────────────────────────────────────────
elif page == "📈 Evaluasi Metode":
    st.title("📈 Evaluasi & Perbandingan Metode")
    st.markdown("---")

    # Coverage
    st.subheader("📊 Coverage Perbandingan Metode")

    apriori_prods = set()
    for _, row in rules.iterrows():
        for item in row["antecedents"]:
            apriori_prods.add(item)
    cb_prods = set(cb_sim_df.index.tolist())
    hybrid_prods = apriori_prods.union(cb_prods)
    all_prods = set(df_clean["Description"].unique())

    coverage_data = pd.DataFrame({
        "Metode": ["Apriori", "Content-Based", "Hybrid"],
        "Coverage (%)": [
            len(apriori_prods)/len(all_prods)*100,
            len(cb_prods)/len(all_prods)*100,
            len(hybrid_prods)/len(all_prods)*100
        ],
        "Jumlah Produk": [len(apriori_prods), len(cb_prods), len(hybrid_prods)]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            coverage_data, x="Metode", y="Coverage (%)",
            color="Metode", text="Coverage (%)",
            color_discrete_sequence=["steelblue","mediumseagreen","coral"],
            title="Coverage per Metode"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, ylim=[0, 115])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(coverage_data, use_container_width=True)
        st.info(f"""
        **📌 Insight Coverage:**
        - Apriori hanya cover **{len(apriori_prods)/len(all_prods)*100:.1f}%** produk
        - **{len(all_prods) - len(apriori_prods)} produk** tidak bisa direkomendasikan dengan Apriori saja
        - Hybrid menutupi gap ini dengan menambahkan Content-Based
        """)

    st.markdown("---")

    # Venn Diagram
    st.subheader("🔵 Overlap Rekomendasi")
    selected_eval = st.selectbox(
        "Pilih produk untuk evaluasi overlap",
        sorted(df_clean["Description"].unique().tolist()),
        key="eval_product"
    )

    set_apr = set(apriori_only(selected_eval, rules, 10)["Produk Rekomendasi"].tolist())
    set_cb = set(cb_only(selected_eval, cb_sim_df, 10)["Produk Rekomendasi"].tolist())

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        venn2([set_apr, set_cb],
              set_labels=("Apriori", "Content-Based"),
              set_colors=("steelblue", "mediumseagreen"),
              alpha=0.6, ax=ax)
        ax.set_title(f"Overlap: {selected_eval[:30]}")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.metric("Apriori rekomendasi", len(set_apr))
        st.metric("CB rekomendasi", len(set_cb))
        st.metric("Overlap (sama)", len(set_apr & set_cb))
        if len(set_apr & set_cb) == 0:
            st.success("✅ Tidak ada overlap → kedua metode saling melengkapi!")
        else:
            st.info(f"Produk yang direkomendasikan keduanya: {', '.join(list(set_apr & set_cb))}")

    st.markdown("---")

    # Distribusi metrik
    st.subheader("📉 Distribusi Metrik Association Rules")
    col1, col2, col3 = st.columns(3)

    for col, metric, color in zip(
        [col1, col2, col3],
        ["support", "confidence", "lift"],
        ["steelblue", "mediumseagreen", "coral"]
    ):
        fig = px.histogram(
            rules, x=metric, nbins=30,
            title=f"Distribusi {metric.capitalize()}",
            color_discrete_sequence=[color]
        )
        fig.add_vline(x=rules[metric].mean(), line_dash="dash",
                      line_color="red",
                      annotation_text=f"Mean: {rules[metric].mean():.3f}")
        col.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# HALAMAN 6: IMPLIKASI MANAJERIAL
# ─────────────────────────────────────────
elif page == "💼 Implikasi Manajerial":
    st.title("💼 Implikasi Manajerial")
    st.markdown("---")

    # Temuan utama
    st.subheader("🔍 Temuan Utama")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error(f"""
        **🛒 Pola Pembelian Kuat**
        - {len(rules)} association rules ditemukan
        - Lift tertinggi: **{rules["lift"].max():.2f}x**
        - Confidence tertinggi: **{rules["confidence"].max():.0%}**
        - Ada pasangan produk yang hampir **selalu dibeli bersama**
        """)

    with col2:
        apriori_prods = set()
        for _, row in rules.iterrows():
            for item in row["antecedents"]:
                apriori_prods.add(item)
        all_prods = set(df_clean["Description"].unique())
        gap = len(all_prods) - len(apriori_prods)
        st.warning(f"""
        **📦 Gap Rekomendasi**
        - Apriori hanya cover **{len(apriori_prods)/len(all_prods)*100:.0f}%** produk
        - **{gap} produk** tidak pernah direkomendasikan
        - Sistem Hybrid menutup gap ini ke **100% coverage**
        """)

    with col3:
        monthly = df_clean.groupby("Month")["InvoiceNo"].nunique()
        peak_m = monthly.idxmax()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily = df_clean["DayOfWeek"].value_counts().reindex(day_order)
        peak_d = daily.idxmax()
        st.info(f"""
        **📅 Pola Waktu**
        - Transaksi tertinggi: **{peak_m}**
        - Hari tersibuk: **{peak_d}**
        - Jam tersibuk: **10:00 - 12:00**
        - Indikasi segmen **B2B/wholesale** dominan
        """)

    st.markdown("---")

    # Rekomendasi aksi
    st.subheader("🎯 Rekomendasi Aksi Bisnis")

    aksi = pd.DataFrame({
        "Temuan": [
            f"Rules dengan lift > 10 ditemukan",
            f"{gap} produk tidak tercover Apriori",
            "Lonjakan transaksi di November",
            "25% transaksi tanpa CustomerID",
            "Jam tersibuk 10:00-12:00 (jam kerja)"
        ],
        "Aksi yang Disarankan": [
            "Buat paket bundle produk seri yang sama dengan diskon kecil",
            "Implementasi sistem Hybrid recommendation di platform",
            "Siapkan kampanye bundle produk mulai Oktober",
            "Buat sistem rekomendasi untuk guest user berbasis popularitas",
            "Kirim email/notifikasi rekomendasi di pagi hari Selasa-Kamis"
        ],
        "Prioritas": [
            "🔴 Tinggi",
            "🔴 Tinggi",
            "🟡 Sedang",
            "🟡 Sedang",
            "🟢 Rendah"
        ],
        "Estimasi Dampak": [
            "Peningkatan average order value",
            "Peningkatan product discovery",
            "Peningkatan revenue Q4",
            "Peningkatan konversi guest user",
            "Peningkatan open rate komunikasi"
        ]
    })

    st.dataframe(aksi, use_container_width=True)

    st.markdown("---")

    # Simulasi dampak bisnis
    st.subheader("💰 Simulasi Dampak Bisnis")
    st.caption("Estimasi kasar berdasarkan asumsi industri e-commerce")

    col1, col2, col3 = st.columns(3)
    avg_order = col1.number_input("Rata-rata nilai order (Rp)", value=500000, step=50000)
    daily_transactions = col2.number_input("Transaksi per hari saat ini", value=100, step=10)
    conversion_rate = col3.slider("Estimasi conversion rate rekomendasi (%)", 1, 30, 10)

    additional_revenue = daily_transactions * (conversion_rate/100) * avg_order
    monthly_revenue = additional_revenue * 30
    yearly_revenue = additional_revenue * 365

    col1, col2, col3 = st.columns(3)
    col1.metric("Tambahan Revenue/Hari", f"Rp {additional_revenue:,.0f}")
    col2.metric("Tambahan Revenue/Bulan", f"Rp {monthly_revenue:,.0f}")
    col3.metric("Tambahan Revenue/Tahun", f"Rp {yearly_revenue:,.0f}")

    st.caption("*Simulasi ini bersifat ilustratif. Hasil aktual bergantung pada implementasi dan kondisi bisnis.")

    st.markdown("---")

    # Top rules untuk eksekutif
    st.subheader("🏆 Top 5 Peluang Cross-Selling")
    top5 = rules.head(5).copy()
    top5["antecedents"] = top5["antecedents"].apply(lambda x: ", ".join(list(x)))
    top5["consequents"] = top5["consequents"].apply(lambda x: ", ".join(list(x)))

    for i, row in top5.iterrows():
        st.success(f"""
        **#{i+1}** Pembeli **{row["antecedents"]}**
        → disarankan membeli **{row["consequents"]}**
        | Probabilitas: **{row["confidence"]*100:.1f}%** | Kekuatan: **{row["lift"]:.1f}x** dari kebetulan
        """)
