import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Rating Prediction – Ensemble Learning Demo",
    layout="centered"
)

st.title("🎬 Rating Prediction with Ensemble Learning")
st.markdown(
    """
    Mục tiêu demo
    So sánh hiệu quả giữa:
    - Baseline model  
    - Mô hình đơn (Base models)  
    - Mô hình tổ hợp (Ensemble learning)  

    Thước đo đánh giá: RMSE (Root Mean Squared Error)  
    RMSE càng thấp → mô hình càng tốt.
    """
)

baseline_df = pd.read_csv("baseline_mean_result.csv")
knn_rf_df = pd.read_csv("knn_rf_sklearn_result.csv")
tree_gb_df = pd.read_csv("DecisionTree_GradientBoosting_results.csv")

baseline_df = baseline_df[["Model", "RMSE", "MAE"]]
knn_rf_df = knn_rf_df[["Model", "RMSE", "MAE"]]
tree_gb_df = tree_gb_df[["Model", "RMSE", "MAE"]]


all_results = pd.concat(
    [baseline_df, knn_rf_df, tree_gb_df],
    ignore_index=True
)


st.subheader("📊 Bảng tổng hợp kết quả các mô hình")
st.dataframe(all_results, use_container_width=True)


st.subheader("📉 So sánh RMSE giữa các mô hình")

fig1, ax1 = plt.subplots(figsize=(8, 5))
bars = ax1.bar(
    all_results["Model"],
    all_results["RMSE"],
    color="skyblue"
)

ax1.set_ylabel("RMSE")
ax1.set_xlabel("Model")
ax1.set_title("RMSE Comparison of All Models")
ax1.set_xticklabels(all_results["Model"], rotation=30, ha="right")

for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax1.grid(axis="y", linestyle="--", alpha=0.5)
st.pyplot(fig1)


st.subheader("🏆 Xếp hạng mô hình theo RMSE (tốt → kém)")

ranking = all_results.sort_values("RMSE")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.barh(
    ranking["Model"],
    ranking["RMSE"],
    color="steelblue"
)

ax2.invert_yaxis()
ax2.set_xlabel("RMSE")
ax2.set_title("Model Ranking by RMSE")

for i, v in enumerate(ranking["RMSE"]):
    ax2.text(
        v + 0.002,
        i,
        f"{v:.3f}",
        va="center",
        fontsize=9
    )

ax2.grid(axis="x", linestyle="--", alpha=0.5)
st.pyplot(fig2)


best_model = ranking.iloc[0]

st.subheader(" Kết luận")

st.markdown(
    f"""
    - Mô hình có RMSE thấp nhất là {best_model['Model']}
    - Điều này cho thấy phương pháp **ensemble / boosting** giúp cải thiện độ chính xác
      so với baseline và các mô hình đơn.
    - Kết quả thực nghiệm khẳng định rằng kết hợp nhiều mô hình**
      giúp giảm sai số và tăng độ ổn định trong bài toán dự đoán rating.
    """
)

st.info(
    "Demo này minh họa kết quả thực nghiệm sau huấn luyện, không thực hiện train lại mô hình."
)
