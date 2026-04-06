"""
SEA Development Analysis — Data Exploration & Preprocessing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = r"E:\Document\Visualization\lab2\datafile\loc_sea_data.csv"
OUTPUT_DIR = r"E:\Document\Visualization\lab2\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print("=== RAW DATA ===")
print(f"Shape     : {df.shape}")
print(f"Columns   : {list(df.columns)}")
print(f"Countries : {df['Country Name'].nunique()} — {sorted(df['Country Name'].unique())}")
print(f"Years     : {int(df['Year'].min())} – {int(df['Year'].max())}")
print(f"Indicators: {df['Series Name'].nunique()}")
print()
print(df.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 2. PIVOT WIDE
# ══════════════════════════════════════════════════════════════════════════════
col_rename = {
    "GDP per capita (constant 2015 US$)":    "GDP_pc",
    "GDP growth (annual %)":                 "GDP_growth",
    "Inflation CPI (annual %)":              "Inflation",
    "Unemployment rate (%)":                 "Unemployment",
    "FDI net inflows (% GDP)":               "FDI",
    "Exports (% GDP)":                       "Exports",
    "Gross capital formation (% GDP)":       "Capital_form",
    "Gov final consumption (% GDP)":         "Gov_spend",
    "Life expectancy at birth":              "Life_expect",
    "Urban population (%)":                  "Urban_pct",
    "Total population":                      "Population",
    "Education expenditure (% GDP)":         "Education",
    "Health expenditure (% GDP)":            "Health",
    "Infant mortality rate":                 "Infant_mort",
    "Renewable energy (% total final energy)": "Renewable",
    "Internet users (% population)":         "Internet",
}

wide = (
    df.pivot_table(index=["Country Name", "Year"],
                   columns="Series Name", values="Value")
      .reset_index()
      .rename(columns=col_rename)
)
wide.columns.name = None
indicator_cols = list(col_rename.values())

print("\n=== WIDE FORMAT ===")
print(f"Shape: {wide.shape}")
print(wide.dtypes.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== MISSING VALUE ANALYSIS ===")

# Tổng thể theo indicator
miss = wide[indicator_cols].isna().mean().mul(100).round(1).sort_values(ascending=False)
print("\nMissing % per indicator:")
print(miss.to_string())

# Theo quốc gia
miss_country = wide.groupby("Country Name")[indicator_cols].apply(
    lambda g: g.isna().mean().mul(100).mean()
).round(1).sort_values(ascending=False)
print("\nAvg missing % per country:")
print(miss_country.to_string())

# Theo năm (để xem giai đoạn nào có ít data)
miss_year = wide.groupby("Year")[indicator_cols].apply(
    lambda g: g.isna().mean().mul(100).mean()
).round(1)
print("\nAvg missing % per year:")
print(miss_year.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== DESCRIPTIVE STATISTICS (all years, all countries) ===")
desc = wide[indicator_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc["count"] = desc["count"].astype(int)
print(desc.round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. SNAPSHOT CHECK — 2019 (pre-COVID reference year)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SNAPSHOT 2019 ===")
snap19 = wide[wide["Year"] == 2019][["Country Name"] + indicator_cols].set_index("Country Name")
coverage_19 = snap19.notna().sum(axis=1).rename("non_null_cols")
print(f"Countries with data in 2019: {snap19.shape[0]}")
print(f"\nNon-null indicator count per country:")
print(coverage_19.to_string())
print(f"\nIndicators with full coverage (11/11 countries):")
full = snap19.columns[snap19.notna().all()].tolist()
print(full)
print(f"\nIndicators with partial coverage (<9 countries):")
partial = snap19.columns[snap19.notna().sum() < 9].tolist()
print(partial if partial else "None")


# ── Notes ───────────────────────────────────────────────────────────────
# 2024 — bỏ hẳn
#   37.5% missing, WB chưa cập nhật đủ. Không có giá trị phân tích.
#
# 2023 — giữ nhưng cẩn thận
#   11.9% missing chủ yếu là Education + Exports/Capital/Gov.
#   Dùng được cho trend nhưng không dùng làm snapshot chính.
#   Snapshot chính nên là 2022 hoặc 2019.
#
# Myanmar — giữ nhưng xử lý riêng
#   27.2% do sau đảo chính 2021, WB ngừng nhận báo cáo.
#   Education: gần như missing toàn bộ → bỏ khỏi feature set
#   Các chỉ số kinh tế sau 2021: missing → forward-fill từ 2020
#
# Tóm lại:
#   1. Bỏ năm 2024 hoàn toàn
#   2. Bỏ cột Education khỏi indicator set (29% missing, Myanmar = 0)
#   3. Năm làm việc: 2000–2023
#   4. Imputation:
#      - Population, Urban_pct, Life_expect, Infant_mort → forward-fill (trending data)
#      - Exports, Capital_form, Gov_spend, Renewable → fillna(median theo từng nước)
#      - GDP_pc, GDP_growth, Inflation, FDI → giữ nguyên, thiếu đâu note đó
#   5. Snapshot cho clustering: dùng 2019 (pre-COVID, coverage đầy đủ nhất)
# ─────────────────────────────────────────────────────────────────────────────
# Dân số (Population) — bỏ khỏi feature set
# Trong dataset này có 11 nước với dân số rất chênh lệch:

# Indonesia: ~270 triệu
# Singapore: ~5.5 triệu
# Brunei: ~0.4 triệu
# Khi đưa Population vào clustering hoặc feature importance, model sẽ nhóm nước theo quy mô địa lý, không theo trình độ phát triển.

# Ví dụ: Indonesia và Lao sẽ bị tách xa nhau chỉ vì dân số khác nhau 50 lần — dù cả hai đều là nước đang phát triển với GDP/capita tương đương.

# ══════════════════════════════════════════════════════════════════════════════
# 6. CLEANING
# Lý do từng quyết định:
#   - Bỏ 2024: 37.5% missing, WB chưa cập nhật
#   - Bỏ Education: 29.1% missing toàn bộ, Myanmar hầu như 0% (sau đảo chính 2021)
#   - Forward-fill trending vars: giá trị thay đổi liên tục theo thời gian
#     (dân số, tuổi thọ...) → giá trị liền trước là ước tính tốt nhất
#   - Median-fill structural vars: tỉ lệ % GDP ít biến động mạnh,
#     median theo nước giữ được đặc trưng riêng từng nước
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 6: CLEANING ===")

# 6a. Bỏ 2024
wide = wide[wide["Year"] <= 2023].copy()
print(f"After dropping 2024: {wide.shape[0]} rows, years {wide['Year'].min()}–{wide['Year'].max()}")

# 6b. Bỏ Education
DROP_COLS = ["Education", "Population"]   # Education: missing quá nhiều; Population: proxy variable
indicator_cols = [c for c in indicator_cols if c not in DROP_COLS]
wide = wide.drop(columns=DROP_COLS, errors="ignore")
print(f"After dropping {DROP_COLS}: {len(indicator_cols)} indicators remain")
print(f"  Indicators: {indicator_cols}")

# 6c. Imputation theo nhóm
FFILL_COLS   = ["Life_expect", "Infant_mort", "Urban_pct"]   # trending — forward then back fill
MEDIAN_COLS  = ["Exports", "Capital_form", "Gov_spend", "Renewable",
                "Health", "Internet"]                          # structural % — median per country
# Còn lại (GDP_pc, GDP_growth, Inflation, FDI, Unemployment) → giữ nguyên

wide = wide.sort_values(["Country Name", "Year"])

for col in FFILL_COLS:
    wide[col] = wide.groupby("Country Name")[col].transform(
        lambda s: s.ffill().bfill()
    )

for col in MEDIAN_COLS:
    wide[col] = wide.groupby("Country Name")[col].transform(
        lambda s: s.fillna(s.median())
    )

miss_after = wide[indicator_cols].isna().mean().mul(100).round(1).sort_values(ascending=False)
print("\nMissing % after imputation:")
print(miss_after[miss_after > 0].to_string() if miss_after.max() > 0 else "  → No missing values!")

# ══════════════════════════════════════════════════════════════════════════════
# 7. NORMALIZATION STRATEGY
#
# Chọn: RobustScaler (median + IQR)  →  không dùng PCA  →  Feature selection
#
# Tại sao RobustScaler thay vì StandardScaler?
#   StandardScaler dùng mean + std — bị kéo lệch bởi outlier:
#     GDP_pc  : Singapore 68k kéo mean lên, Myanmar 300 bị ép về -2.x std
#     Inflation: Myanmar 57% sau coup kéo mean, làm các nước bình thường xấp xỉ nhau
#     Exports : Singapore 229% GDP là outlier cực đoan
#   RobustScaler dùng median + IQR → outlier không ảnh hưởng đến scale
#   → Khoảng cách giữa các nước ở nhóm giữa (VN, TH, PH...) được phân biệt tốt hơn
#
# Tại sao không dùng PCA?
#   PCA yêu cầu dữ liệu zero-mean → StandardScaler mới đảm bảo, RobustScaler thì không
#   Ngoài ra: dataset chỉ có 11 nước × 14 features
#     → nhiều feature hơn sample (n_features > n_samples)
#     → PCA trong không gian này kém ý nghĩa, dễ overfit cấu trúc noise
#   Kết luận: không cần PCA khi số quan sát quá nhỏ
#
# Dùng gì thay thế PCA để giảm chiều?
#   Feature selection: chọn thủ công 6-8 chỉ số đại diện cho từng nhóm
#     (kinh tế, xã hội, môi trường) → clustering có ý nghĩa kinh tế hơn
#   Visualize: dùng scatter matrix hoặc MDS 2D sau khi cluster
#
# Indicator nào cần log-transform trước khi scale?
#   GDP_pc: phân phối lệch phải rất mạnh (min 299, max 68k) → log làm phẳng
#   Exports: Singapore 229% là outlier cực đoan trong nhóm % GDP → log giảm ảnh hưởng
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 7: NORMALIZATION ===")

# Log-transform các cột lệch phải trước khi scale
wide["GDP_pc_log"] = np.log1p(wide["GDP_pc"])
wide["Exports_log"] = np.log1p(wide["Exports"])
log_indicator_cols = [
    "GDP_pc_log" if c == "GDP_pc" else
    "Exports_log" if c == "Exports" else c
    for c in indicator_cols
]

# Snapshot 2019 để demo scale
snap19_clean = wide[wide["Year"] == 2019].copy().reset_index(drop=True)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(snap19_clean[log_indicator_cols].fillna(
    snap19_clean[log_indicator_cols].median()
))

print(f"Scaled matrix shape (2019 snapshot): {X_scaled.shape}")
print(f"  Median per feature (should be ~0): {np.median(X_scaled, axis=0).round(2)}")

scaled_df = pd.DataFrame(X_scaled, columns=log_indicator_cols,
                         index=snap19_clean["Country Name"].values)
print("\nScaled values (2019 snapshot):")
print(scaled_df.round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY — ready for analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== DATA READY ===")
print(f"  wide          : {wide.shape}  (cleaned, imputed, 2000-2023)")
print(f"  indicators    : {len(indicator_cols)} cols → {indicator_cols}")
print(f"  log-transform : GDP_pc → GDP_pc_log,  Exports → Exports_log")
print(f"  scaler        : RobustScaler (median+IQR, outlier-robust)")
print(f"  dim reduction : Feature selection (không dùng PCA — n_samples < n_features)")
print(f"  Next step     : Feature selection → K-Means → Cluster profiling → COVID analysis")


# ── Notes — Quan sát từ scaled output (2019 snapshot) ──────────────────
#
# [1] Outlier thực tế, không phải lỗi data:
#   - Singapore  FDI        = 11.96  → trung tâm tài chính, FDI/GDP thuộc hàng cao nhất TG
#   - Timor-Leste Gov_spend = 7.14   → chi tiêu từ Petroleum Fund (dầu mỏ)
#   - Timor-Leste GDP_growth= 6.63   → tăng trưởng bất thường từ khai thác dầu
#   - Timor-Leste FDI       = -4.77  → FDI âm: vốn rút ra nhiều hơn đầu vào
#   - Myanmar    Inflation  = 3.09   → khủng hoảng kinh tế sau đảo chính 2021
#   → Không xóa các điểm này, chúng phản ánh đặc thù kinh tế của từng nước
#
# [2] Phân nhóm sơ bộ từ GDP_pc_log:
#   Cao    : Singapore (2.17), Brunei (1.63), Malaysia (0.85)
#   Trung  : Thailand (0.45), Indonesia (0.06), Philippines (0.00)
#   Thấp   : Viet Nam (-0.08), Lao (-0.26), Cambodia (-0.44)
#   Rất thấp: Timor-Leste (-0.66), Myanmar (-0.69)
#   → K=3 hoặc K=4 là hợp lý — cần validate bằng silhouette
#
# [3] Feature selection — chọn 8 chỉ số đại diện (tránh redundancy):
#   Kinh tế  (3): GDP_pc_log, GDP_growth, FDI
#     bỏ Exports    — correlated với FDI (trung tâm thương mại = FDI cao)
#     bỏ Inflation  — nhiễu ngắn hạn, không đặc trưng phát triển
#     bỏ Unemployment — nhiễu ngắn hạn
#   Xã hội   (3): Life_expect, Internet, Health
#     bỏ Infant_mort — nghịch chiều Life_expect → redundant
#     bỏ Urban_pct   — correlated với Internet
#   Cấu trúc (2): Capital_form, Renewable
#     đại diện đầu tư dài hạn + định hướng phát triển bền vững
#     bỏ Gov_spend   — Timor-Leste outlier 7.14 làm nhiễu clustering
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 8: FEATURE SELECTION ===")

FEATURE_COLS = [
    "GDP_pc_log",    # kinh tế — mức độ giàu có
    "GDP_growth",    # kinh tế — tốc độ tăng trưởng
    "FDI",           # kinh tế — mức độ hội nhập
    "Life_expect",   # xã hội  — chất lượng sống tổng hợp
    "Internet",      # xã hội  — mức độ số hóa
    "Health",        # xã hội  — đầu tư y tế
    "Capital_form",  # cấu trúc — đầu tư dài hạn
    "Renewable",     # cấu trúc — định hướng phát triển bền vững
]

# Kiểm tra correlation giữa các feature đã chọn (để xác nhận ít redundancy)
snap19_feats = snap19_clean[
    ["GDP_pc_log" if c == "GDP_pc" else
     "Exports_log" if c == "Exports" else c
     for c in indicator_cols]
].copy()
snap19_feats.columns = [
    "GDP_pc_log" if c == "GDP_pc" else
    "Exports_log" if c == "Exports" else c
    for c in indicator_cols
]

corr = snap19_feats[FEATURE_COLS].corr().round(2)
print("Correlation matrix (8 selected features, 2019):")
print(corr.to_string())

# Cặp nào corr > 0.85 → xem xét lại
corr_vals = corr.to_numpy()
high_corr = []
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        val = abs(float(corr_vals[i, j]))
        if val > 0.85:
            high_corr.append((FEATURE_COLS[i], FEATURE_COLS[j], round(val, 2)))
if high_corr:
    print(f"\nHigh correlation pairs (>0.85): {high_corr}")
    print("  → Cân nhắc bỏ một trong mỗi cặp")
else:
    print("\nNo high-correlation pairs (>0.85) — feature set OK")

# ══════════════════════════════════════════════════════════════════════════════
# 9. K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 9: K-MEANS CLUSTERING ===")

# Scale chỉ 8 features đã chọn
snap19_feat_vals = snap19_clean[FEATURE_COLS].fillna(snap19_clean[FEATURE_COLS].median())
scaler_km = RobustScaler()
X_km = scaler_km.fit_transform(snap19_feat_vals)
countries_19 = snap19_clean["Country Name"].values

# 9a. Chọn K tối ưu bằng silhouette score
K_range = range(2, 6)
sil_scores = {}
for k in K_range:
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    labels = km.fit_predict(X_km)
    score = silhouette_score(X_km, labels)
    sil_scores[k] = round(score, 4)
    print(f"  K={k}: silhouette = {score:.4f}")

best_k = max(sil_scores, key=lambda k: sil_scores[k])
print(f"\n  => Best K = {best_k} (silhouette = {sil_scores[best_k]})")

# 9b. Fit final model
km_final = KMeans(n_clusters=best_k, n_init=50, random_state=42)
labels_final = km_final.fit_predict(X_km)

snap19_clean = snap19_clean.copy()
snap19_clean["Cluster"] = labels_final.tolist()

print(f"\nCluster composition (K={best_k}, 2019 snapshot):")
for cl in sorted(snap19_clean["Cluster"].unique()):
    members = snap19_clean[snap19_clean["Cluster"] == cl]["Country Name"].tolist()
    gdp_avg = snap19_clean[snap19_clean["Cluster"] == cl]["GDP_pc"].mean()
    print(f"  Cluster {cl}: {members}  |  avg GDP/cap = ${gdp_avg:,.0f}")

# 9c. Silhouette per sample — xem nước nào "nằm sai cluster"
sil_vals = np.array(silhouette_samples(X_km, labels_final))
snap19_clean["Silhouette"] = pd.Series(sil_vals, index=snap19_clean.index)
print(f"\nSilhouette score per country:")
print(
    snap19_clean[["Country Name", "Cluster", "Silhouette"]]
    .sort_values("Silhouette")
    .to_string(index=False)
)
low_sil = snap19_clean[snap19_clean["Silhouette"] < 0.1]
if not low_sil.empty:
    print(f"\nCountries with silhouette < 0.1 (borderline):")
    print(low_sil[["Country Name", "Cluster", "Silhouette"]].to_string(index=False))

# ── Visualize: Silhouette plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
colors = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B"]
for cl in range(best_k):
    cl_sil = np.sort(sil_vals[labels_final == cl])
    size = cl_sil.shape[0]
    y_upper = y_lower + size
    ax.barh(range(y_lower, y_upper), cl_sil, height=1.0,
            color=colors[cl], alpha=0.8, label=f"Cluster {cl}")
    ax.text(-0.05, y_lower + size / 2, f"C{cl}", fontsize=9, va="center")
    y_lower = y_upper + 5

ax.axvline(sil_scores[best_k], color="red", ls="--", lw=1.5,
           label=f"Avg = {sil_scores[best_k]:.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_title(f"Silhouette Plot — K={best_k} (2019 snapshot)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_01_silhouette.png"), bbox_inches="tight")
plt.show()
print(f"  → Saved: fig_01_silhouette.png")

# ── Visualize: Cluster heatmap (mean raw values per cluster) ──────────────
profile_raw = snap19_clean.groupby("Cluster")[FEATURE_COLS].mean()
profile_scaled = pd.DataFrame(
    scaler_km.transform(profile_raw),
    columns=FEATURE_COLS,
    index=profile_raw.index
)

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(profile_scaled, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Scaled value (RobustScaler)"})
ax.set_title(f"Cluster Profiles — Mean Feature Values (K={best_k}, 2019)")
ax.set_ylabel("Cluster")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_02_cluster_profile.png"), bbox_inches="tight")
plt.show()
print(f"  → Saved: fig_02_cluster_profile.png")


# ── Notes (Lộc) — Kết quả Step 8 & 9, quyết định hướng đi ───────────────────
#
# [Step 8] Feature redundancy:
#   GDP_pc_log vs Life_expect : corr = 0.89
#   GDP_pc_log vs Internet    : corr = 0.90
#   Life_expect vs Internet   : corr = 0.87
#   → Cả ba đo cùng một chiều: "mức độ phát triển chung"
#   → Giữ lại chỉ GDP_pc_log, bỏ Life_expect và Internet khỏi feature set
#   → Feature set mới (6 chỉ số):
#       GDP_pc_log, GDP_growth, FDI, Health, Capital_form, Renewable
#
# [Step 9] Singapore isolation:
#   K=2 silhouette = 0.618 — cao nhưng vô nghĩa:
#     Cluster 1 = [Singapore] duy nhất, GDP/cap $61,346
#     Cluster 0 = 10 nước còn lại, GDP/cap $6,551
#   K-Means chỉ "học" được Singapore khác biệt, không có insight gì thêm
#   Singapore silhouette = 0.0 → nó không thuộc về cluster nào rõ ràng
#
# [Quyết định] Chọn Hướng A:
#   - Loại Singapore khỏi clustering
#   - Cluster 10 nước còn lại → kỳ vọng ra 2-3 nhóm có ý nghĩa
#   - Singapore giữ lại như "reference benchmark" (đích đến phát triển)
#   - Lý do: insight rõ ràng hơn cho báo cáo, tránh outlier chi phối kết quả
#
# [Next step]
#   1. Cập nhật FEATURE_COLS: bỏ Life_expect, Internet
#   2. Lọc Singapore ra trước khi scale + cluster
#   3. Chạy lại K-Means trên 10 nước với 6 features
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 10. REVISED CLUSTERING — 6 features, Singapore excluded
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 10: REVISED CLUSTERING (ex-Singapore) ===")

FEATURE_COLS_V2 = [
    "GDP_pc_log",   # mức độ phát triển kinh tế
    "GDP_growth",   # tốc độ tăng trưởng
    "FDI",          # mức độ hội nhập
    "Health",       # đầu tư xã hội
    "Capital_form", # đầu tư dài hạn
    "Renewable",    # định hướng phát triển bền vững
]

# Tách Singapore ra làm benchmark
snap19_sgp  = snap19_clean[snap19_clean["Country Name"] == "Singapore"].copy()
snap19_10   = snap19_clean[snap19_clean["Country Name"] != "Singapore"].copy().reset_index(drop=True)

print(f"Clustering on {len(snap19_10)} countries (Singapore excluded as benchmark)")
print(f"Features: {FEATURE_COLS_V2}")

# Scale
feat_10 = snap19_10[FEATURE_COLS_V2].fillna(snap19_10[FEATURE_COLS_V2].median())
scaler_v2 = RobustScaler()
X_10 = scaler_v2.fit_transform(feat_10)
countries_10 = snap19_10["Country Name"].values

# Chọn K
K_range = range(2, 6)
sil_scores_v2 = {}
for k in K_range:
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    lbl = km.fit_predict(X_10)
    score = silhouette_score(X_10, lbl)
    sil_scores_v2[k] = round(score, 4)
    print(f"  K={k}: silhouette = {score:.4f}")

best_k_v2 = max(sil_scores_v2, key=lambda k: sil_scores_v2[k])
print(f"\n  => Best K = {best_k_v2} (silhouette = {sil_scores_v2[best_k_v2]})")

# Fit final
km_v2 = KMeans(n_clusters=best_k_v2, n_init=50, random_state=42)
labels_v2 = km_v2.fit_predict(X_10)
snap19_10["Cluster"] = labels_v2.tolist()

print(f"\nCluster composition (K={best_k_v2}, ex-Singapore, 2019):")
for cl in sorted(snap19_10["Cluster"].unique()):
    members  = snap19_10[snap19_10["Cluster"] == cl]["Country Name"].tolist()
    gdp_avg  = snap19_10[snap19_10["Cluster"] == cl]["GDP_pc"].mean()
    grow_avg = snap19_10[snap19_10["Cluster"] == cl]["GDP_growth"].mean()
    print(f"  Cluster {cl}: {members}")
    print(f"    avg GDP/cap = ${gdp_avg:,.0f}  |  avg GDP growth = {grow_avg:.1f}%")

# Silhouette per country
sil_v2 = np.array(silhouette_samples(X_10, labels_v2))
snap19_10["Silhouette"] = pd.Series(sil_v2, index=snap19_10.index)
print(f"\nSilhouette per country:")
print(
    snap19_10[["Country Name", "Cluster", "Silhouette"]]
    .sort_values(["Cluster", "Silhouette"], ascending=[True, False])
    .to_string(index=False)
)
borderline = snap19_10[snap19_10["Silhouette"] < 0.1]
if not borderline.empty:
    print(f"\nBorderline countries (silhouette < 0.1): {borderline['Country Name'].tolist()}")

# Singapore benchmark
sgp_vals = snap19_sgp[FEATURE_COLS_V2].fillna(0).values
sgp_scaled = scaler_v2.transform(sgp_vals)
print(f"\nSingapore (benchmark, not clustered):")
print(pd.DataFrame(sgp_scaled, columns=FEATURE_COLS_V2, index=["Singapore"]).round(2).to_string())

# ── Notes ─────────────────────────────────────────────────────────────────────
# Timor-Leste isolated (silhouette=0.0) vì GDP_growth = 24.2% — doanh thu dầu mỏ,
# không đại diện năng lực phát triển thực. Cùng pattern với Singapore.
# → Hướng A: bỏ cả Singapore + Timor-Leste, cluster 9 nước còn lại
#   Ghi chú báo cáo: "Singapore (financial hub) và Timor-Leste (oil microstate)
#   được xử lý như special cases do cấu trúc kinh tế không đại diện cho SEA"
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 11. FINAL CLUSTERING — 9 countries, Singapore + Timor-Leste excluded
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 11: FINAL CLUSTERING (9 countries) ===")

SPECIAL_CASES = ["Singapore", "Timor-Leste"]
snap19_main    = snap19_clean[~snap19_clean["Country Name"].isin(SPECIAL_CASES)].copy().reset_index(drop=True)
snap19_special = snap19_clean[snap19_clean["Country Name"].isin(SPECIAL_CASES)].copy()

print(f"Main cluster: {snap19_main['Country Name'].tolist()}")
print(f"Special cases (excluded): {SPECIAL_CASES}")

# Scale
feat_main = snap19_main[FEATURE_COLS_V2].fillna(snap19_main[FEATURE_COLS_V2].median())
scaler_final = RobustScaler()
X_main = scaler_final.fit_transform(feat_main)
countries_main = snap19_main["Country Name"].values

# Chọn K
sil_final = {}
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    lbl = km.fit_predict(X_main)
    score = silhouette_score(X_main, lbl)
    sil_final[k] = round(score, 4)
    print(f"  K={k}: silhouette = {score:.4f}")

best_k_final = max(sil_final, key=lambda k: sil_final[k])
print(f"\n  => Best K = {best_k_final} (silhouette = {sil_final[best_k_final]})")

# Fit
km_final2 = KMeans(n_clusters=best_k_final, n_init=50, random_state=42)
labels_final2 = km_final2.fit_predict(X_main)
snap19_main["Cluster"] = labels_final2.tolist()

print(f"\nCluster composition (K={best_k_final}, 2019):")
for cl in sorted(snap19_main["Cluster"].unique()):
    members  = snap19_main[snap19_main["Cluster"] == cl]["Country Name"].tolist()
    gdp_avg  = snap19_main[snap19_main["Cluster"] == cl]["GDP_pc"].mean()
    grow_avg = snap19_main[snap19_main["Cluster"] == cl]["GDP_growth"].mean()
    fdi_avg  = snap19_main[snap19_main["Cluster"] == cl]["FDI"].mean()
    print(f"  Cluster {cl}: {members}")
    print(f"    GDP/cap=${gdp_avg:,.0f}  |  growth={grow_avg:.1f}%  |  FDI={fdi_avg:.1f}%")

# Silhouette per country
sil_arr = np.array(silhouette_samples(X_main, labels_final2))
snap19_main["Silhouette"] = pd.Series(sil_arr, index=snap19_main.index)
print(f"\nSilhouette per country:")
print(
    snap19_main[["Country Name", "Cluster", "Silhouette"]]
    .sort_values(["Cluster", "Silhouette"], ascending=[True, False])
    .to_string(index=False)
)

# Special cases — project lên cùng scale để so sánh
special_vals = snap19_special[FEATURE_COLS_V2].fillna(snap19_special[FEATURE_COLS_V2].median())
special_scaled = scaler_final.transform(special_vals)
print(f"\nSpecial cases (scaled relative to 9-country space):")
print(pd.DataFrame(special_scaled, columns=FEATURE_COLS_V2,
                   index=snap19_special["Country Name"].values).round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# KẾT LUẬN — Tại sao dừng Clustering, chuyển hướng phân tích
# ══════════════════════════════════════════════════════════════════════════════
#
# Quá trình thử nghiệm (Step 9 → 10 → 11):
#   Step 9  (11 nước, 8 features): Singapore isolated  → silhouette 0.618 (giả cao)
#   Step 10 (10 nước, 6 features): Timor-Leste isolated → silhouette 0.633 (giả cao)
#   Step 11 ( 9 nước, 6 features): Cambodia isolated   → silhouette 0.388
#                                   Viet Nam borderline 0.108
#                                   Brunei + Myanmar cùng cluster (vô nghĩa)
#
# Nguyên nhân:
#   - Mỗi nước SEA là một "structural outlier" theo cách riêng:
#       Singapore   = financial hub (FDI/GDP thuộc hàng cao nhất TG)
#       Timor-Leste = oil microstate (GDP_growth bị bóp méo bởi dầu mỏ)
#       Cambodia    = garment manufacturing hub (FDI/GDP = 10%)
#       Brunei      = energy exporter (GDP/cap ~$30k nhưng tăng trưởng thấp)
#   - n_samples (9-11) quá nhỏ so với n_features (6-8)
#     → K-Means không đủ dữ liệu để tìm cấu trúc thực sự
#   - SEA không có "cụm tự nhiên" ở snapshot 2019 — 11 nước = 11 con đường
#     phát triển khác nhau
#
# Kết luận kỹ thuật:
#   Clustering không sai về mặt kỹ thuật, nhưng câu hỏi nghiên cứu
#   không phù hợp với phương pháp này trên dataset này.
#   Silhouette cao = outlier bị tách, không phải cluster có ý nghĩa.
#
# Hướng phân tích thay thế (phù hợp hơn với câu hỏi nghiên cứu):
#   1. Correlation analysis  — chỉ số nào kéo chỉ số nào
#   2. COVID impact analysis — nước nào bị ảnh hưởng nặng / phục hồi tốt nhất
#   3. Trajectory analysis   — ai tăng trưởng nhanh nhất 2000–2023
#
# File này (sea_clustering_modeling.py) giữ lại làm tài liệu quá trình.
# Phân tích tiếp theo thực hiện trong lab2.ipynb.
# ══════════════════════════════════════════════════════════════════════════════
