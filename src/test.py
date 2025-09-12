import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Dữ liệu mới cần dự đoán
# Đây là dữ liệu bạn đã cung cấp trong câu hỏi đầu tiên.
new_data_dict = {
    'MDVP:Fo(Hz)': 101.16105074156427, 'MDVP:Fhi(Hz)': 107.23523710872166,
    'MDVP:Flo(Hz)': 95.16111694540162, 'MDVP:Jitter(%)': 0.00772622386679588,
    'MDVP:Jitter(Abs)': 7.63422261861713e-05, 'MDVP:RAP': 0.004311720505676734,
    'MDVP:PPQ': 0.004354179803098736, 'Jitter:DDP': 0.0129351615170302,
    'MDVP:Shimmer': 0.10876856472086908, 'MDVP:Shimmer(dB)': 1.0068113601785469,
    'Shimmer:APQ3': 0.050810227831560424, 'Shimmer:APQ5': 0.07132802017632987,
    'MDVP:APQ': 0.07580923654892252, 'Shimmer:DDA': 0.15243068349468128,
    'HNR': 9.86270925669202, 'NHR': 0.10139201855935098,
    'DFA': 1.146744862740653, 'D2': -1.4777884376123935e-15,
    'spread1': 98.39004219248412, 'spread2': 1.0192117119521864,
    'PPE': 1.99819672195696, 'mean_MFCC_0th_coef': -262.14975,
    'mean_MFCC_1st_coef': 85.78633, 'mean_MFCC_2nd_coef': -28.194029,
    'mean_MFCC_3rd_coef': 7.915591, 'mean_MFCC_4th_coef': -8.512299,
    'mean_MFCC_5th_coef': 4.6707726, 'mean_MFCC_6th_coef': -4.0398927,
    'mean_MFCC_7th_coef': -5.1903377, 'mean_MFCC_8th_coef': -10.129794,
    'mean_MFCC_9th_coef': -1.9336158, 'mean_MFCC_10th_coef': -8.353839,
    'mean_MFCC_11th_coef': -7.091504, 'mean_MFCC_12th_coef': -11.357413
}

# 2. Tải và chuẩn bị dữ liệu huấn luyện từ file CSV
# Thay 'extracted_features.csv' bằng đường dẫn đúng tới file của bạn
try:
    df = pd.read_csv('./data/extracted_features.csv')

    # Tách dữ liệu thành các đặc trưng (X) và nhãn (y)
    # Chúng ta loại bỏ cột 'id' và 'class' khỏi các đặc trưng
    X_train = df.drop(['id', 'class'], axis=1)
    y_train = df['class']

    # Lấy thứ tự các cột đặc trưng để đảm bảo dữ liệu mới có cùng thứ tự
    feature_order = X_train.columns

    # Sắp xếp lại dữ liệu mới theo đúng thứ tự của dữ liệu huấn luyện
    new_data_list = [new_data_dict[feature] for feature in feature_order]
    new_data_array = np.array(new_data_list).reshape(1, -1)

    # 3. Chuẩn hóa dữ liệu
    # Việc này rất quan trọng để mô hình hoạt động hiệu quả
    scaler = StandardScaler()

    # Dùng dữ liệu huấn luyện (X_train) để "học" cách chuẩn hóa
    X_train_scaled = scaler.fit_transform(X_train)

    # Áp dụng cách chuẩn hóa vừa học vào dữ liệu mới
    new_data_scaled = scaler.transform(new_data_array)

    # 4. Huấn luyện mô hình học máy
    # Sử dụng RandomForestClassifier, một mô hình mạnh mẽ và phổ biến
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_train_scaled, y_train)

    # 5. Đưa ra dự đoán trên dữ liệu mới
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)

    # 6. In kết quả
    print("--- Kết Quả Phân Tích Giọng Nói ---")
    print(f"Nhãn dự đoán: {prediction[0]}")

    if prediction[0] == 1:
        print("Diễn giải: Mẫu giọng có dấu hiệu của bệnh Parkinson.")
    else:
        print("Diễn giải: Mẫu giọng bình thường (khỏe mạnh).")

    print("\nXác suất dự đoán cho các lớp:")
    print(f" - Lớp 0 (Khỏe mạnh): {probability[0][0]:.2%}")
    print(f" - Lớp 1 (Parkinson): {probability[0][1]:.2%}")
    print("\nLưu ý: Đây không phải là chẩn đoán y tế.")


except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'extracted_features.csv'. Vui lòng kiểm tra lại tên và đường dẫn file.")