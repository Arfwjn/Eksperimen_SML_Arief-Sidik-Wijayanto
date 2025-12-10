import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import joblib

#  PATH 
INPUT_PATH = './examScorePrediction_raw/Exam_Score_Prediction.csv'
# Path penyimpanan output preprocessing
OUTPUT_DIR = 'preprocessing/examScorePrediction_preprocessing/'
PROCESSED_DATA_PATH = OUTPUT_DIR + 'data_preprocessing.csv'
PREPROCESSOR_PATH = OUTPUT_DIR + 'preprocessor.pkl'
KMEANS_MODEL_PATH = OUTPUT_DIR + 'kmeans_model.pkl'

# K untuk K-Means
K_CLUSTERS = 4 

# 1. LOAD & CLEAN DATA RAW
print("--- 1. LOAD & CLEAN DATA RAW ---")
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

# 2. DEFINISI & FIT COLUMN TRANSFORMER
print("\n--- 2. DEFINISI & FIT COLUMN TRANSFORMER ---")
numerical_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
nominal_cols = ['gender', 'course', 'internet_access', 'study_method']
ordinal_cols = ['sleep_quality', 'facility_rating', 'exam_difficulty']

ordinal_categories = [
    ['poor', 'average', 'good'], 
    ['low', 'medium', 'high'], 
    ['easy', 'moderate', 'hard']
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

# FIT preprocessor pada seluruh fitur X
X_transformed = preprocessor.fit_transform(X_features)
print(f"Dimensi Fitur setelah Preprocessing: {X_transformed.shape}")

# Simpan objek preprocessor
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"Preprocessor disimpan di: {PREPROCESSOR_PATH}")


# 3. K-MEANS CLUSTERING PADA FITUR X
print("\n3. K-MEANS CLUSTERING & LABELING")

# K-Means diterapkan pada fitur X yang sudah distandarisasi
kmeans = KMeans(n_clusters=K_CLUSTERS, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_transformed)

# Simpan model K-Means
joblib.dump(kmeans, KMEANS_MODEL_PATH)
print(f"K-Means Model disimpan di: {KMEANS_MODEL_PATH}")

# Gabungkan hasil cluster dengan target Y untuk mapping
X_transformed['cluster_label'] = kmeans.labels_
X_transformed['exam_score'] = y_target_raw.values

# Mapping: Urutkan cluster berdasarkan rata-rata exam_score (rendah ke tinggi)
cluster_means = X_transformed.groupby('cluster_label')['exam_score'].mean().sort_values()
cluster_map = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}

# Mapping untuk menghasilkan label klasifikasi
y = X_transformed['cluster_label'].map(cluster_map).astype(int)

X = X_transformed.drop(columns=['cluster_label', 'exam_score']).copy()
X.reset_index(drop=True, inplace=True) 

# Statistik Kelas Hasil K-Means
print(f"\nStatistik Kelas Hasil K-Means (K={K_CLUSTERS}):")
bin_stats = df_processed.groupby(y)['exam_score'].agg(['min', 'mean', 'max', 'count'])
print(bin_stats.to_markdown(numalign='left', stralign='left'))


# 4. TRAIN-TEST SPLIT & SAVING OUTPUT
print("\n4. TRAIN-TEST SPLIT & SAVING OUTPUT")
X['target'] = y

X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['target']), 
    X['target'].values,
    test_size=0.2, 
    random_state=42, 
    stratify=X['target'].values
)

y_train = pd.Series(y_train, index=X_train.index)
y_test = pd.Series(y_test, index=X_test.index)

df_final_features = pd.concat([X_train, X_test])

df_final_target = pd.concat([y_train, y_test])

df_final = df_final_features.copy()
df_final['target'] = df_final_target
df_final['is_train'] = df_final.index.isin(X_train.index)

# Simpan hasil
df_final.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Data final (X dan y) disimpan di: {PROCESSED_DATA_PATH}")

print("\nCHECK SHAPES & CLASS DISTRIBUTION:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Proporsi Kelas Target di y_train:\n{y_train.value_counts(normalize=True).sort_index().to_markdown(numalign='left', stralign='left')}")