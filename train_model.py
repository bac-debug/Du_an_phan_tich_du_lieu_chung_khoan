# File: train_model.py (PHIÊN BẢN NÂNG CẤP)

import os
import pandas as pd
import yfinance as yf
import joblib
from ml_model import train_xgb_model_by_timeframe
from features.feature_engineering import them_chi_bao_ky_thuat, add_technical_features

# ==========================
# CẤU HÌNH
# ==========================
TICKERS = ["VIC", "VCB", "HPG", "MSN", "VHM", "CTG", "TCB", "GAS", "VRE", "PLX"]
TIMEFRAMES = ["short", "medium", "long"]
SAVE_DIR = os.path.join(os.path.dirname(__file__), "../models")

# ==========================
# HÀM CHÍNH
# ==========================
# File: train_model.py

def main():
    """
    Hàm chính để tải dữ liệu, xử lý và huấn luyện mô hình cho NHIỀU mã cổ phiếu.
    Phiên bản này xử lý lỗi cho từng mã một cách độc lập.
    """
    print(f"🚀 Bắt đầu quá trình huấn luyện cho các mã: {', '.join(TICKERS)}")
    
    # --- Vòng lặp chính để xử lý TỪNG MÃ CỔ PHIẾU ---
    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"🧠 Bắt đầu xử lý mã: {ticker}")
        print(f"{'='*60}")

        # Tạo thư mục lưu trữ riêng cho từng mã
        ticker_save_dir = os.path.join(SAVE_DIR, ticker)
        os.makedirs(ticker_save_dir, exist_ok=True)
        
        # --- Bước 1: Tải dữ liệu ---
        print(f"\n🔄 Đang tải dữ liệu cho {ticker} (2 năm gần nhất)...")
        try:
            data = yf.download(f"{ticker}.VN", period="2y", interval="1d", progress=False)
            if data.empty:
                data = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            if data.empty:
                print(f"❌ Không thể tải dữ liệu cho mã {ticker}. Bỏ qua mã này.")
                continue

            print(f"✅ Đã tải thành công {len(data)} ngày dữ liệu cho {ticker}.")
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu cho {ticker}: {e}. Bỏ qua mã này.")
            continue

        # === SỬA LỖI QUAN TRỌNG TẠI ĐÂY ===
        # Xử lý trường hợp yfinance trả về cột đa cấp (MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            # Giữ lại cấp độ đầu tiên ('Open', 'Close',...) và loại bỏ cấp độ thứ hai
            data.columns = data.columns.get_level_values(0)
        
        # Bây giờ, tất cả tên cột đều là string đơn giản, có thể chuyển sang chữ thường an toàn
        data.columns = [str(col).lower() for col in data.columns]
        
        # Ưu tiên sử dụng giá đã điều chỉnh nếu có
        if 'adj close' in data.columns:
            data = data.rename(columns={'adj close': 'close'})

        # --- Bước 2: Thêm các đặc trưng kỹ thuật ---
        print("\n🔧 Đang xử lý và thêm các đặc trưng kỹ thuật...")
        data_with_indicators = them_chi_bao_ky_thuat(data)
        full_features_data = add_technical_features(data_with_indicators)
        print("✅ Đã thêm đầy đủ các đặc trưng.")

        # --- Bước 3: Huấn luyện và lưu mô hình cho từng khung thời gian ---
        print("\n💪 Bắt đầu huấn luyện cho các khung thời gian...")
        for tf in TIMEFRAMES:
            print(f"\n--- Đang huấn luyện mô hình '{tf}' cho {ticker} ---")
            
            model, metrics, features = train_xgb_model_by_timeframe(full_features_data, tf)
            
            save_path = os.path.join(ticker_save_dir, f"model_{tf}.pkl")
            joblib.dump((model, features), save_path)
            
            print(f"✅ Đã lưu mô hình tại: {save_path}")
            print(f"   📊 Kết quả: RMSE = {metrics['rmse']:.2f} | MAPE = {metrics['mape']:.2f}%")

    print(f"\n{'='*60}")
    print("🎉 Hoàn tất quá trình huấn luyện cho tất cả các mã có thể xử lý!")
if __name__ == "__main__":
    main()