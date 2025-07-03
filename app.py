import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="VaporIQ Analytics Dashboard")

@st.cache_data
def load_data():
    # Adjust path to Data folder
    return pd.read_csv('Data/vaporiq_synthetic_dataset_10k.csv')

data = load_data()

tabs = st.tabs(["Data Visualization","Classification","Clustering","Association Rules","Regression"])

# 1. Data Visualization
with tabs[0]:
    st.header("Data Visualization")
    st.write("### 10+ Descriptive Complex Insights")
    fig, ax = plt.subplots()
    data['age'].hist(bins=20, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    # Additional insights would follow similarly

# 2. Classification
with tabs[1]:
    st.header("Classification Models")
    features = st.multiselect("Select feature columns", options=data.select_dtypes(include=[np.number]).columns.tolist(), default=['age','income','monthly_vape_spend'])
    target = 'willingness_to_subscribe'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    st.table(pd.DataFrame(results))

    sel = st.selectbox("Select model for confusion matrix", options=list(models.keys()))
    cm = confusion_matrix(y_test, models[sel].predict(X_test))
    st.write("Confusion Matrix for", sel)
    st.write(cm)

    fig, ax = plt.subplots()
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    st.write("### Predict New Data")
    uploaded = st.file_uploader("Upload CSV without target", type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        preds = models['Random Forest'].predict(new_df[features])
        new_df['prediction'] = preds
        st.write(new_df)
        csv = new_df.to_csv(index=False).encode()
        st.download_button("Download Predictions", data=csv, file_name='predictions.csv')

# 3. Clustering
with tabs[2]:
    st.header("Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 4)
    numeric = data.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(numeric)
    data['cluster'] = kmeans.labels_

    distortions = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42).fit(numeric)
        distortions.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

    st.write("### Cluster Personas")
    persona = data.groupby('cluster').mean()[['age','income','monthly_vape_spend']].round(2)
    st.write(persona)

    csv = data.to_csv(index=False).encode()
    st.download_button("Download Clustered Data", data=csv, file_name='clustered_data.csv')

# 4. Association Rules
with tabs[3]:
    st.header("Association Rule Mining")
    cols = st.multiselect("Select transaction columns", options=['liked_flavors','disliked_flavors'], default=['liked_flavors'])
    minsup = st.number_input("Min Support", min_value=0.01, max_value=1.0, value=0.05)
    minconf = st.number_input("Min Confidence", min_value=0.01, max_value=1.0, value=0.3)
    trans = data[cols].apply(lambda x: x.str.get_dummies(sep=','))
    freq = apriori(trans, min_support=minsup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=minconf).sort_values('confidence', ascending=False).head(10)
    st.write(rules[['antecedents','consequents','support','confidence','lift']])

# 5. Regression
with tabs[4]:
    st.header("Regression Models")
    reg_features = st.multiselect("Regression features", options=data.select_dtypes(include=[np.number]).columns.tolist(), default=['income','age','usage_freq_per_week'])
    target_reg = st.selectbox("Regression target", options=['monthly_vape_spend','satisfaction_rating'])
    Xr = data[reg_features]
    yr = data[target_reg]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)

    regs = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor()
    }
    reg_results = []
    for name, model in regs.items():
        model.fit(Xr_train, yr_train)
        score = model.score(Xr_test, yr_test)
        reg_results.append({'Model': name, 'R2': score})
    st.table(pd.DataFrame(reg_results))