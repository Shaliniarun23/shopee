import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules
from lifetimes import BetaGeoFitter, GammaGammaFitter

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'customer_id': range(1, n+1),
        'total_orders': np.random.poisson(5, n),
        'avg_order_value': np.round(np.random.uniform(20, 200, n), 2),
        'days_since_last_order': np.random.randint(1, 60, n),
        'app_opens': np.random.randint(1, 100, n),
        'session_duration': np.round(np.random.uniform(1, 30, n), 1),
        'cart_abandons': np.random.randint(0, 10, n),
        'converted': np.random.choice([0, 1], size=n)
    })
    # Synthetic transaction data
    tx_data = pd.DataFrame([
        {'transaction_id': i, **{item: np.random.choice([0,1], p=[0.7,0.3])
         for item in ['Electronics','Clothing','Home','Beauty','Toys']}}
        for i in range(1, 1001)
    ])
    return df, tx_data

df, tx = load_data()

st.set_page_config(layout="wide", page_title="Shopee Analytics Dashboard")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sales Forecast", "Customer Clustering", "Churn Prediction", "CLV Prediction", "Association Rules"])

# ------------------------- HOME -------------------------
if page == "Home":
    st.title("📊 Shopee Analytics Dashboard")
    st.markdown("Use the sidebar to navigate through each analysis module.")
    st.markdown("Modules include:")
    st.markdown("- Sales Forecasting (Linear Regression)")
    st.markdown("- Customer Clustering (K-Means)")
    st.markdown("- Churn Prediction (Classification)")
    st.markdown("- Customer Lifetime Value (CLV)")
    st.markdown("- Association Rule Mining")

# ------------------------- SALES FORECAST -------------------------
elif page == "Sales Forecast":
    st.title("📈 Sales Forecasting")
    st.markdown("Using Linear Regression to predict Sales")

    xvar = st.selectbox("Select X Variable", ['app_opens', 'session_duration', 'cart_abandons'])
    y = df['total_orders']
    X = df[[xvar]]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    fig = px.scatter(df, x=xvar, y='total_orders', trendline="ols", labels={xvar: xvar, 'total_orders': 'Orders'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regression Coefficient:")
    st.write(f"{xvar}: {round(model.coef_[0], 2)}, Intercept: {round(model.intercept_, 2)}")

# ------------------------- CUSTOMER CLUSTERING -------------------------
elif page == "Customer Clustering":
    st.title("🧠 Customer Segmentation with K-Means")
    features = ['total_orders', 'avg_order_value', 'days_since_last_order']
    k = st.slider("Select Number of Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=42).fit(df[features])
    df['cluster'] = km.labels_

    xvar = st.selectbox("X-Axis", features, key="clust_x")
    yvar = st.selectbox("Y-Axis", features, key="clust_y")
    fig = px.scatter(df, x=xvar, y=yvar, color=df['cluster'].astype(str), labels={'color':'Cluster'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Counts")
    st.write(df['cluster'].value_counts())

# ------------------------- CHURN PREDICTION -------------------------
elif page == "Churn Prediction":
    st.title("❌ Churn Prediction (Converted vs. Not)")
    X = df[['app_opens', 'session_duration', 'cart_abandons']]
    y = df['converted']
    model = GradientBoostingClassifier().fit(X, y)
    pred = model.predict(X)

    st.subheader("Classification Report:")
    report = classification_report(y, pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# ------------------------- CLV -------------------------
elif page == "CLV Prediction":
    st.title("💸 Customer Lifetime Value (CLV) Prediction")

    clv_df = df[df['total_orders'] > 0].copy()
    clv_df['frequency'] = clv_df['total_orders'] - 1
    clv_df['recency'] = clv_df['days_since_last_order']
    clv_df['monetary'] = clv_df['avg_order_value']
    clv_df['T'] = 60  # Constant observation period

    # Drop missing or invalid rows
    clv_df = clv_df.dropna(subset=['frequency', 'recency', 'monetary'])
    clv_df = clv_df[clv_df['frequency'] >= 0]

    st.markdown(f"Modeling on {len(clv_df)} customers with >1 order")

    try:
        bgf = BetaGeoFitter()
        bgf.fit(clv_df['frequency'], clv_df['recency'], clv_df['T'])
        ggf = GammaGammaFitter()
        ggf.fit(clv_df['frequency'], clv_df['monetary'])

        horizon = st.slider("Forecast horizon (days):", 30, 180, 90, step=30)
        clv = bgf.customer_lifetime_value(ggf, clv_df['frequency'], clv_df['recency'], clv_df['T'],
                                          clv_df['monetary'], time=horizon)
        st.subheader("Top 10 Customers by CLV")
        st.dataframe(clv.sort_values(ascending=False).head(10))
    except Exception as e:
        st.error(f"Error in CLV Modeling: {e}")

# ------------------------- ASSOCIATION RULES -------------------------
elif page == "Association Rules":
    st.title("🔗 Association Rule Mining")

    # Elbow Curve
    st.subheader("Elbow Curve: Support vs Frequent Itemsets")
    supports = np.linspace(0.01, 0.1, 10)
    counts = [len(apriori(tx.drop(columns=['transaction_id']), min_support=s, use_colnames=True)) for s in supports]
    elbow = pd.DataFrame({'Support': supports, 'Frequent Itemsets': counts})
    st.plotly_chart(px.line(elbow, x='Support', y='Frequent Itemsets'), use_container_width=True)

    st.subheader("Clustered Association Patterns")
    cluster_k = st.slider("Number of Clusters for Pattern Personas", 2, 10, 4)

    min_sup = st.slider("Min Support:", 0.01, 0.1, 0.05)
    min_conf = st.slider("Min Confidence:", 0.1, 1.0, 0.3)
    basket = tx.drop(columns=['transaction_id'])
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        st.subheader("Top 10 Association Rules by Lift (Hypersona Table)")
        top_rules = rules.sort_values('lift', ascending=False).head(10)
        st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        st.warning("No rules found for selected thresholds.")
