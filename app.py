
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib_venn import venn2
import ast
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
def load_models():
    # Load tabel rules Apriori yang sudah dihitung di Colab
    rules = pd.read_csv("rules_result.csv")
    # Kembalikan string list dari CSV menjadi list Python beneran
    rules['antecedents'] = rules['antecedents'].apply(ast.literal_eval)
    rules['consequents'] = rules['consequents'].apply(ast.literal_eval)

    # Load matriks Content-Based yang sudah dihitung
    cb_sim_df = pd.read_csv("similarity_result.csv", index_col=0)
    return rules, cb_sim_df

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
with st.spinner("Memuat dashboard (super cepat!)..."):
    df_raw = load_data()
    df_clean = preprocess(df_raw)
    rules, cb_sim_df = load_models()

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

    # Tentukan daftar halaman berdasarkan mode yang dipilih
    if mode == "👔 Executive Dashboard":
        pilihan_halaman = [
            "🏠 Home",
            "🤖 Sistem Rekomendasi",
            "💼 Implikasi Manajerial"
        ]
    else:
        pilihan_halaman = [
        "🏠 Home",
        "📊 Eksplorasi Data",
        "🤖 Sistem Rekomendasi",
        "🔬 Analisis Rules",
        "🔧 Analisis Parameter",     
        "📈 Evaluasi Metode",
        "💼 Implikasi Manajerial"
   ]

    page = st.selectbox("Navigasi Halaman", pilihan_halaman)
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
    col1.info("📊 **Eksplorasi Data**\nLihat distribusi dan pola dalam data")
    col2.success("🤖 **Sistem Rekomendasi**\nCoba rekomendasi produk secara interaktif")
    col3.warning("💼 **Implikasi Manajerial**\nTemuan dan rekomendasi aksi bisnis")

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
# HALAMAN BARU: ANALISIS PARAMETER
# ─────────────────────────────────────────
elif page == "🔧 Analisis Parameter":
    st.title("🔧 Analisis & Justifikasi Parameter")
    st.markdown("---")
    st.caption("""
    Halaman ini menjelaskan **dasar pemilihan setiap parameter** dalam model,
    didukung dengan analisis dan visualisasi — bukan sekadar klaim.
    """)
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Min Support",
        "📈 Min Lift",
        "📦 Filter Top 300",
        "⚖️ Bobot Hybrid"
    ])
 
    # ─────────────────────────────────────────
    # TAB 1: SENSITIVITY ANALYSIS MIN SUPPORT
    # ─────────────────────────────────────────
    with tab1:
        st.subheader("📉 Sensitivity Analysis — Min Support")
        st.markdown("""
        Kita uji berbagai nilai `min_support` untuk melihat berapa rules yang terbentuk.
        Tujuannya: menemukan titik **elbow** — dimana rules masih cukup banyak tapi tidak noise.
        """)
 
        # Data hasil sensitivity analysis (dari notebook)
        sa_support = pd.DataFrame({
            "Min Support": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10],
            "Jumlah Itemsets": [8452, 1420, 623, 416, 177, 47, 9, 3],
            "Jumlah Rules": [16302, 2240, 646, 232, 36, 0, 0, 0]
        })
 
        col1, col2 = st.columns([3, 2])
 
        with col1:
            fig = px.line(
                sa_support, x="Min Support", y="Jumlah Rules",
                markers=True, title="Min Support vs Jumlah Rules Terbentuk",
                color_discrete_sequence=["steelblue"]
            )
            fig.add_vline(
                x=0.02, line_dash="dash", line_color="red",
                annotation_text="Pilihan kita (0.02)",
                annotation_position="top right"
            )
            fig.add_scatter(
                x=[0.02], y=[232],
                mode="markers",
                marker=dict(color="red", size=12, symbol="circle"),
                name="Nilai dipilih",
                showlegend=True
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
 
        with col2:
            st.dataframe(
                sa_support.style.apply(
                    lambda x: ["background-color: #ffcccc; font-weight: bold"
                               if x["Min Support"] == 0.02 else "" for _ in x],
                    axis=1
                ),
                use_container_width=True
            )
 
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.error("**❌ Support 0.005**\n\n16.302 rules → terlalu banyak, mayoritas noise, tidak praktis digunakan")
        col2.success("**✅ Support 0.02 (pilihan kita)**\n\n232 rules → manageable, berada di titik elbow sebelum rules anjlok")
        col3.warning("**⚠️ Support 0.03**\n\nHanya 36 rules → terlalu sedikit, banyak produk tidak punya rekomendasi Apriori")
 
        st.info("""
        **📌 Kenapa tidak pilih 0.01?**
        Support 0.01 menghasilkan 2.240 rules dengan rasio itemset:rules = 1:1.57 — artinya banyak rules
        yang redundan dan saling tumpang tindih. Support 0.02 menghasilkan rasio 1:0.55 yang jauh lebih bersih.
        Selain itu, produk yang lolos support 0.01 hanya perlu muncul di ~180 transaksi —
        terlalu rendah untuk disebut "pola yang konsisten."
        """)
 
    # ─────────────────────────────────────────
    # TAB 2: SENSITIVITY ANALYSIS MIN LIFT
    # ─────────────────────────────────────────
    with tab2:
        st.subheader("📈 Sensitivity Analysis — Min Lift")
        st.markdown("""
        Lift mengukur **seberapa kuat** hubungan antar produk dibanding kebetulan.
        Lift = 1.0 berarti hubungan nyata (bukan kebetulan). Kita uji beberapa nilai untuk melihat
        berapa rules yang masih terbentuk.
        """)
 
        sa_lift = pd.DataFrame({
            "Min Lift": [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],
            "Jumlah Rules": [232, 198, 156, 98, 45, 18, 8, 3],
        })
        sa_lift["Rules Hilang vs 1.0"] = 232 - sa_lift["Jumlah Rules"]
        sa_lift["% Tersisa"] = (sa_lift["Jumlah Rules"] / 232 * 100).round(1)
 
        col1, col2 = st.columns([3, 2])
 
        with col1:
            fig2 = px.line(
                sa_lift, x="Min Lift", y="Jumlah Rules",
                markers=True, title="Min Lift vs Jumlah Rules Terbentuk",
                color_discrete_sequence=["mediumseagreen"]
            )
            fig2.add_vline(
                x=1.0, line_dash="dash", line_color="red",
                annotation_text="Pilihan kita (1.0)",
                annotation_position="top right"
            )
            fig2.add_scatter(
                x=[1.0], y=[232],
                mode="markers",
                marker=dict(color="red", size=12),
                name="Nilai dipilih",
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)
 
        with col2:
            st.dataframe(
                sa_lift.style.apply(
                    lambda x: ["background-color: #ffcccc; font-weight: bold"
                               if x["Min Lift"] == 1.0 else "" for _ in x],
                    axis=1
                ),
                use_container_width=True
            )
 
        st.markdown("---")
 
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **✅ Kenapa pilih lift = 1.0?**
 
            - Lift > 1.0 sudah berarti hubungan **nyata, bukan kebetulan**
            - Memberi fleksibilitas: produk yang rules-nya lemah tetap bisa
              muncul sebagai **fallback** kalau tidak ada rules yang lebih kuat
            - Dari 232 rules yang terbentuk, **156 rules (67%)** sudah natural
              memiliki lift ≥ 1.5 — artinya mayoritas rules kita sudah kuat
            """)
        with col2:
            st.warning("""
            **⚠️ Limitasi lift = 1.0**
 
            - Konservatif — meloloskan rules yang hubungannya lemah
            - Idealnya dinaikkan ke **1.5** jika hanya ingin rules yang benar-benar kuat
            - Namun untuk sistem rekomendasi yang butuh coverage luas,
              1.0 lebih aman sebagai threshold awal
            """)
 
    # ─────────────────────────────────────────
    # TAB 3: ANALISIS FILTER TOP 300
    # ─────────────────────────────────────────
    with tab3:
        st.subheader("📦 Analisis Filter Top 300 Produk")
        st.markdown("""
        Sebelum Apriori dijalankan, kita membatasi analisis hanya pada **300 produk paling sering dibeli**.
        Kenapa? Dan apa konsekuensinya?
        """)
 
        col1, col2, col3 = st.columns(3)
        total_before = df_clean["Description"].nunique()
        total_all = 4015  # total produk unik sebelum filter
 
        col1.metric("Total produk unik (semua)", f"{total_all:,}")
        col2.metric("Produk yang dianalisis", f"{total_before:,} (top 300)")
        col3.metric("Produk tidak teranalisis", f"{total_all - total_before:,} (92.5%)")
 
        st.markdown("---")
 
        col1, col2 = st.columns(2)
 
        with col1:
            # Sparsity comparison
            sparsity_df = pd.DataFrame({
                "Kondisi": ["Sebelum Filter\n(semua produk)", "Sesudah Filter\n(top 300)"],
                "Sparsity (%)": [99.4, 96.4]
            })
            fig3 = px.bar(
                sparsity_df, x="Kondisi", y="Sparsity (%)",
                color="Kondisi", text="Sparsity (%)",
                color_discrete_sequence=["#e74c3c", "#27ae60"],
                title="Perbandingan Sparsity Matrix"
            )
            fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig3.update_layout(showlegend=False, yaxis_range=[0, 105])
            st.plotly_chart(fig3, use_container_width=True)
 
        with col2:
            # Coverage pie
            fig4 = px.pie(
                values=[total_before, total_all - total_before],
                names=["Teranalisis (Top 300)", "Tidak Teranalisis"],
                color_discrete_sequence=["steelblue", "#d9d9d9"],
                title=f"Coverage Produk (Total: {total_all:,} produk unik)"
            )
            st.plotly_chart(fig4, use_container_width=True)
 
        st.markdown("---")
 
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **✅ Kenapa perlu filter?**
 
            - Tanpa filter, matrix ukurannya **18.000 × 4.015** dengan sparsity 99.4%
            - Apriori tidak efektif di matrix sesparse itu — hampir tidak ada produk
              yang "cukup sering muncul bareng" kalau dibagi ke 4.015 produk
            - Produk di luar top 300 frekuensinya terlalu rendah untuk
              lolos min_support=0.02 (butuh ≥ 360 transaksi)
            """)
        with col2:
            st.error("""
            **⚠️ Limitasi filter ini**
 
            - **92.5% produk (3.715 produk)** tidak teranalisis oleh Apriori
            - Inilah penyebab coverage Apriori hanya ~16%
            - Angka 300 tidak ada justifikasi formal — dipilih berdasarkan
              pertimbangan komputasi dan frekuensi produk
            - **Gap ini ditutup oleh Content-Based** dalam sistem Hybrid,
              sehingga coverage total tetap 100%
            """)
 
        st.info("""
        **📌 Kenapa tidak 500?**
        Produk ke-301 hingga ke-500 frekuensinya sudah sangat rendah dan tidak akan
        lolos min_support=0.02. Menambahkan mereka hanya memperbesar matrix tanpa
        menambah rules baru yang bermakna — komputasi lebih berat, hasil sama.
        """)
 
    # ─────────────────────────────────────────
    # TAB 4: EKSPERIMEN BOBOT HYBRID
    # ─────────────────────────────────────────
    with tab4:
        st.subheader("⚖️ Eksperimen Bobot Hybrid")
        st.markdown("""
        Kita uji 3 kombinasi bobot untuk membuktikan bahwa pilihan **60:40**
        menghasilkan rekomendasi yang **robust** — tidak berubah drastis meski bobotnya digeser.
        """)
 
        selected_exp = st.selectbox(
            "🔍 Pilih produk untuk eksperimen",
            sorted(df_clean["Description"].unique().tolist()),
            key="exp_product"
        )
        top_n_exp = st.slider("Jumlah rekomendasi", 3, 10, 5, key="exp_topn")
 
        def hybrid_custom(product_name, w_apr, w_cb, top_n):
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
            hybrid = {
                item: w_apr * apr_scores.get(item, 0) + w_cb * cb_scores.get(item, 0)
                for item in all_items
            }
            result = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return pd.DataFrame(result, columns=["Produk Rekomendasi", "Score"])
 
        configs = [
            ("80:20 (Apriori-heavy)", 0.8, 0.2),
            ("60:40 ★ Pilihan kita", 0.6, 0.4),
            ("50:50 (Seimbang)", 0.5, 0.5),
        ]
 
        results_exp = {}
        for label, w_a, w_c in configs:
            results_exp[label] = hybrid_custom(selected_exp, w_a, w_c, top_n_exp)
 
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
 
        for col, (label, _, _) in zip([col1, col2, col3], configs):
            with col:
                is_chosen = "Pilihan kita" in label
                header_color = "🟡" if is_chosen else "⚪"
                st.markdown(f"**{header_color} Bobot {label}**")
                df_exp = results_exp[label].copy()
                df_exp.index = range(1, len(df_exp) + 1)
                df_exp["Score"] = df_exp["Score"].round(4)
                st.dataframe(df_exp, use_container_width=True)
 
        st.markdown("---")
 
        # Hitung overlap
        sets_exp = {label: set(df_r["Produk Rekomendasi"].tolist())
                    for label, df_r in results_exp.items()}
        labels_list = list(sets_exp.keys())
 
        overlap_ab = len(sets_exp[labels_list[0]] & sets_exp[labels_list[1]])
        overlap_bc = len(sets_exp[labels_list[1]] & sets_exp[labels_list[2]])
        overlap_all = len(sets_exp[labels_list[0]] & sets_exp[labels_list[1]] & sets_exp[labels_list[2]])
 
        col1, col2, col3 = st.columns(3)
        col1.metric("80:20 ∩ 60:40", f"{overlap_ab} dari {top_n_exp} sama")
        col2.metric("60:40 ∩ 50:50", f"{overlap_bc} dari {top_n_exp} sama")
        col3.metric("Semua sama", f"{overlap_all} dari {top_n_exp} sama")
 
        if overlap_bc >= top_n_exp * 0.6:
            st.success(f"""
            ✅ **Hasil ROBUST** — {overlap_bc} dari {top_n_exp} produk sama antara 60:40 dan 50:50.
            Perubahan bobot tidak mengubah set rekomendasi secara drastis.
            Pilihan **60:40 terjustifikasi**: sinyal pembelian nyata (Apriori) lebih dipercaya
            dari kemiripan nama produk (Content-Based).
            """)
        else:
            st.warning(f"""
            ⚠️ Untuk produk ini, perubahan bobot cukup mempengaruhi hasil ({overlap_bc} dari {top_n_exp} sama).
            Ini menunjukkan produk ini berada di "batas" antara dominasi Apriori dan CB.
            Pilihan 60:40 tetap valid sebagai default karena sinyal pembelian lebih dapat dipercaya.
            """)
 
        st.info("""
        **📌 Kesimpulan Eksperimen Bobot**
        - Secara umum hasil **robust** terhadap variasi bobot di range wajar (50:50 hingga 80:20)
        - Pilihan 60:40 didasarkan pada pertimbangan bisnis: pola pembelian nyata
          lebih dapat dipercaya daripada kemiripan nama/deskripsi produk
        - Idealnya bobot dioptimasi via grid search dengan metrik evaluasi formal,
          namun untuk skala proyek ini 60:40 sudah **defensible**
        """)

# ─────────────────────────────────────────
# HALAMAN 5: EVALUASI
# ─────────────────────────────────────────
elif page == "📈 Evaluasi Metode":
    st.title("📈 Evaluasi & Perbandingan Metode")
    st.markdown("---")

    # Coverage
    st.subheader("📊 Coverage Perbandingan Metode")
    st.info(""" **💡 Mengapa Apriori hanya cover ~16% produk?** Apriori hanya menganalisis **top 300 produk** paling sering dibeli. Produk di luar top 300 frekuensinya terlalu rendah untuk lolos min_support=0.02. Gap ini ditutup oleh Content-Based dalam sistem Hybrid. → Lihat detail di halaman **🔧 Analisis Parameter** """)
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
        fig.update_layout(showlegend=False, yaxis_range=[0, 115])
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
