import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

# Import các mô hình
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

print("--- Bắt đầu quá trình huấn luyện Voting và Stacking ---")

# --- 1. Load dữ liệu ---
try:
    df = pd.read_csv('./data/cleaned_features.csv')
    print("Tải dữ liệu từ thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp.")
    exit()
df = df.dropna()


# Bỏ cột không cần thiết
X = df.drop(columns=['class'])


y = df['class']

# Lưu danh sách feature
feature_columns = X.columns.tolist()
output_dir_voting = './model_assets_voting'
output_dir_stacking = './model_assets_stacking'
os.makedirs(output_dir_voting, exist_ok=True)
os.makedirs(output_dir_stacking, exist_ok=True)

with open(os.path.join(output_dir_voting, 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f)
with open(os.path.join(output_dir_stacking, 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f)

print(f"Đã chọn và lưu {len(feature_columns)} đặc trưng.")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Chuyển về DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Kết hợp lại với y
df_preprocessed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


joblib.dump(scaler, os.path.join(output_dir_voting, 'scaler.joblib'))
joblib.dump(scaler, os.path.join(output_dir_stacking, 'scaler.joblib'))
print("Đã tạo và lưu scaler.")

# --- 3. Base learners ---
clf1 = SVC(probability=True, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = XGBClassifier(eval_metric='logloss', random_state=42)

# --- VotingClassifier ---
voting_model = VotingClassifier(
    estimators=[('svc', clf1), ('rf', clf2), ('xgb', clf3)],
    voting='soft'
)

# --- StackingClassifier ---
meta_clf = LogisticRegression(random_state=42)
stacking_model = StackingClassifier(
    estimators=[('svc', clf1), ('rf', clf2), ('xgb', clf3)],
    final_estimator=meta_clf,
    stack_method="predict_proba",
    n_jobs=-1
)

# --- 4. Train models ---
print("\nHuấn luyện VotingClassifier...")
voting_model.fit(X_train, y_train)
joblib.dump(voting_model, os.path.join(output_dir_voting, 'voting_model.joblib'))

print("Huấn luyện StackingClassifier...")
stacking_model.fit(X_train, y_train)
joblib.dump(stacking_model, os.path.join(output_dir_stacking, 'stacking_model.joblib'))

print("Đã huấn luyện và lưu cả Voting & Stacking.")

# --- 5. Đánh giá ---
def evaluate_model(name, model, X_test, y_test):
    print(f"\n--- Đánh giá {name} ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc

acc_voting = evaluate_model("VotingClassifier", voting_model, X_test, y_test)
acc_stacking = evaluate_model("StackingClassifier", stacking_model, X_test, y_test)

print("\n--- So sánh Kết quả ---")
print(f"VotingClassifier Accuracy : {acc_voting:.4f}")
print(f"StackingClassifier Accuracy: {acc_stacking:.4f}")

print("\n--- Quá trình huấn luyện hoàn tất ---")
