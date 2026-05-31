import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def run_preprocessing():
    print("STATUS: Memulai proses preprocessing otomatis...")
    
    raw_path = '../dataset_raw/student_dropout.csv'
    out_dir = '../dataset_preprocessing'
 
    df = pd.read_csv(raw_path, sep=';')
    
    df.drop_duplicates(inplace=True)
    
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target'])
    
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f'{out_dir}/X_train.csv', index=False)
    X_val.to_csv(f'{out_dir}/X_val.csv', index=False)
    X_test.to_csv(f'{out_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{out_dir}/y_train.csv', index=False)
    y_val.to_csv(f'{out_dir}/y_val.csv', index=False)
    y_test.to_csv(f'{out_dir}/y_test.csv', index=False)
    
    print(f"STATUS: Preprocessing selesai. Data bersih berhasil diekspor ke folder '{out_dir}'.")

if __name__ == "__main__":
    run_preprocessing()