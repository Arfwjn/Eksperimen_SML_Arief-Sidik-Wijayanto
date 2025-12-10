import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

INPUT_PATH = '../examScorePrediction_raw/Exam_Score_Prediction.csv'

# 1. LOAD DATA RAW
df = pd.read_csv(INPUT_PATH)
df_processed = df.drop(columns=['student_id']).copy()

# Capping Outlier pada 'exam_score' 
col_outlier = 'exam_score'
Q1 = df_processed[col_outlier].quantile(0.25)
Q3 = df_processed[col_outlier].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_processed[col_outlier] = df_processed[col_outlier].clip(lower=lower_bound, upper=upper_bound)

# Pisahkan Target (Y) dan Fitur (X)
y_target_raw = df_processed['exam_score']
X_features = df_processed.drop(columns=['exam_score'])


# 2. DEFINISI COLUMN TRANSFORMER (HANYA PADA FITUR X)
numerical_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
nominal_cols = ['gender', 'course', 'internet_access', 'study_method']
ordinal_cols = ['sleep_quality', 'facility_rating', 'exam_difficulty']

ordinal_categories = [
    ['poor', 'average', 'good'], ['low', 'medium', 'high'], ['easy', 'moderate', 'hard']
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), nominal_cols),
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")


# 3. K-MEANS CLUSTERING PADA FITUR X
print("1. Preprocessing Fitur")
X_transformed = preprocessor.fit_transform(X_features)
print(f"Dimensi Fitur setelah Preprocessing: {X_transformed.shape}")

# K-Means diterapkan pada fitur X yang sudah distandarisasi
K = 4
kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_transformed)

# Gabungkan hasil cluster dengan target Y untuk mapping
X_transformed['cluster_label'] = kmeans.labels_
X_transformed['exam_score'] = y_target_raw.values

# Mapping: Urutkan cluster berdasarkan rata-rata exam_score
cluster_means = X_transformed.groupby('cluster_label')['exam_score'].mean().sort_values()
cluster_map = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}

# Mapping untuk menghasilkan label klasifikasi
y = X_transformed['cluster_label'].map(cluster_map).astype(int)

# Hapus kolom sementara dan kolom target dari X
X = X_transformed.drop(columns=['cluster_label', 'exam_score']).copy()
del X_transformed

print("\n--- 2. Statistik Kelas Hasil K-Means pada FITUR X ---")
bin_stats = df_processed.groupby(y)['exam_score'].agg(['min', 'mean', 'max', 'count'])
print(bin_stats.to_markdown(numalign='left', stralign='left'))


# 4. TRAIN-TEST SPLIT
print("\n--- 3. Pembagian Data (Train-Test Split) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Proporsi Kelas Target di y_train:\n{y_train.value_counts(normalize=True).sort_index().to_markdown(numalign='left', stralign='left')}")