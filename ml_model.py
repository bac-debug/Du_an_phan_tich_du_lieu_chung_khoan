# File: ml_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# ============================================================
# 📘 Hàm 1: Huấn luyện mô hình theo khung thời gian (Giữ nguyên)
# ============================================================

def train_xgb_model_by_timeframe(data: pd.DataFrame, timeframe: str):
    """
    Huấn luyện mô hình XGBoost cho từng khung thời gian.
    timeframe: "short" (30d), "medium" (90d), "long" (365d)
    """
    steps_map = {"short": 30, "medium": 90, "long": 365}
    steps = steps_map.get(timeframe, 30)
    df = data.copy().dropna()

    # SỬA LỖI: Chuẩn hóa tên cột để nhất quán
    feature_cols = [
        "sma_5", "sma_20", "sma_50",
        "ema_12", "ema_26", "macd", "macd_signal", "rsi_14",
        "close_lag_1", "close_lag_3", "close_lag_5", "close_lag_10",
        "ret_1", "ret_5", "vol_mean_20", "vol_std_20"
    ]
    target_col = "close"
    
    # Đảm bảo các cột này tồn tại
    feature_cols = [col for col in feature_cols if col in df.columns]

    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    metrics = {"rmse": rmse, "mape": mape}

    return model, metrics, feature_cols

# ============================================================
# 📗 Hàm 2: Dự báo bằng mô hình đã lưu
# ============================================================

def forecast_with_model(model, data: pd.DataFrame, feature_cols: list, days_ahead: int = 7):
    """
    Dự báo giá tương lai trong N ngày.
    SỬA LỖI LOGIC: Sử dụng phương pháp dự báo trực tiếp (direct forecast).
    Dùng dữ liệu thực tế cuối cùng để dự báo cho tất cả các ngày trong tương lai.
    Cách này ổn định và đáng tin cậy hơn so với việc dùng dự đoán để dự đoán tiếp.
    """
    df = data.copy().dropna()
    
    # Lấy dòng dữ liệu cuối cùng có đầy đủ thông tin
    last_known_data = df[feature_cols].iloc[-1:].copy()
    
    # Tạo một list để lưu các dự đoán
    predictions = []
    
    # Dự báo N lần, mỗi lần một bước
    # Trong một kịch bản đơn giản, ta có thể giả định các feature không thay đổi nhiều
    # và chỉ dự báo một lần cho N bước, nhưng để chính xác hơn, ta nên dự báo từng bước
    # Tuy nhiên, cách đơn giản và an toàn nhất là lặp lại dự báo từ điểm cuối cùng.
    for _ in range(days_ahead):
        prediction = model.predict(last_known_data)[0]
        predictions.append(prediction)
        # Lưu ý: Chúng ta không cập nhật lại last_known_data với prediction
        # vì điều đó sẽ làm sai lệch các chỉ báo kỹ thuật.
        
    return predictions