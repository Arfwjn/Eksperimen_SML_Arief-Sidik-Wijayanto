import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import os

# Custom Transformer: K-Means Binning
class KMeansBinning(BaseEstimator, TransformerMixin):
    """
    Mengubah variabel numerik 'exam_score' menjadi label kelas (0, 1, 2, 3) 
    menggunakan K-Means Clustering, kemudian mapping label agar urut.
    """
    def __init__(self, n_clusters=4, target_col='exam_score', random_state=42):
        self.n_clusters = n_clusters
        self.target_col = target_col
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                             max_iter=300, n_init=10, random_state=random_state)
        self.cluster_map = None

    def fit(self, X, y=None):
        X_target = X[[self.target_col]].values
        self.kmeans.fit(X_target)

        # Urutan label berdasarkan mean score (0=rendah, K-1=tinggi)
        X_temp = X.copy()
        X_temp['Score_Bin'] = self.kmeans.labels_
        cluster_means = X_temp.groupby('Score_Bin')[self.target_col].mean().sort_values()
        self.cluster_map = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}
        
        return self

    def transform(self, X):
        X_target = X[[self.target_col]].values
        labels = self.kmeans.predict(X_target)
        X['Score_Bin_Mapped'] = [self.cluster_map[label] for label in labels]
        
        # Hapus kolom target
        X_transformed = X.drop(columns=[self.target_col])
        return X_transformed.reset_index(drop=True)


def run_preprocessing(input_file_path, output_dir_path):
    """
    Fungsi untuk menjalankan pipeline preprocessing.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {input_file_path}")
        return

    # 2. Cleaning dan Outlier Capping
    df_processed = df.drop(columns=['student_id'])
    
    # Capping Outlier pada 'exam_score'
    col_outlier = 'exam_score'
    Q1 = df_processed[col_outlier].quantile(0.25)
    Q3 = df_processed[col_outlier].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col_outlier] = df_processed[col_outlier].clip(lower=lower_bound, upper=upper_bound)


    # 3. Encoding Kategori
    numerical_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    nominal_cols = ['gender', 'course', 'internet_access', 'study_method']
    ordinal_cols = ['sleep_quality', 'facility_rating', 'exam_difficulty']
    
    # Kategori Ordinal 
    ordinal_categories = [
        ['poor', 'average', 'good'], # sleep_quality
        ['low', 'medium', 'high'],   # facility_rating
        ['easy', 'moderate', 'hard'] # exam_difficulty
    ]

    # 4. Column Transformer (untuk Scaling dan Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), nominal_cols),
            ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols) 
        ],
        remainder='passthrough', # skip target column
        verbose_feature_names_out=False
    ).set_output(transform="pandas")


    # 5. Preprocessor + Binning    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('binning', KMeansBinning(n_clusters=4, target_col='exam_score'))
    ])

    # 6. Eksekusi Pipeline
    df_transformed = full_pipeline.fit_transform(df_processed)

    # 7. Simpan Hasil
    output_file_path = os.path.join(output_dir_path, 'data_preprocessing.csv')
    os.makedirs(output_dir_path, exist_ok=True)
    df_transformed.to_csv(output_file_path, index=False)
        
    print(f"Dataset hasil preprocessing ({df_transformed.shape}) disimpan di: {output_file_path}")
    
    return df_transformed

if __name__ == '__main__':
    # Logika penentuan PATH 
    if os.path.basename(os.getcwd()) == 'preprocessing':
        INPUT_PATH = '../examScorePrediction_raw/Exam_Score_Prediction.csv'
        OUTPUT_PATH = './examScorePrediction_preprocessing'    
    
    run_preprocessing(INPUT_PATH, OUTPUT_PATH)