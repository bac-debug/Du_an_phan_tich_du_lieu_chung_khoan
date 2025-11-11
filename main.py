import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def tai_du_lieu_chung_khoan(ma_ck, ngay_bat_dau="2020-01-01", ngay_ket_thuc=None):
    """
    Tải dữ liệu chứng khoán từ Yahoo Finance
    
    Args:
        ma_ck (str): Mã chứng khoán (VD: VCB, VIC, HPG)
        ngay_bat_dau (str): Ngày bắt đầu (format: YYYY-MM-DD)
        ngay_ket_thuc (str): Ngày kết thúc (format: YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: Dữ liệu giá chứng khoán
    """
    if ngay_ket_thuc is None:
        ngay_ket_thuc = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Đang tải dữ liệu cho mã {ma_ck} từ {ngay_bat_dau} đến {ngay_ket_thuc}...")
    
    try:
        # Thử tải với .VN suffix trước
        ticker = f"{ma_ck}.VN"
        data = yf.download(ticker, start=ngay_bat_dau, end=ngay_ket_thuc, progress=False)
        
        if data.empty:
            # Fallback về mã gốc
            data = yf.download(ma_ck, start=ngay_bat_dau, end=ngay_ket_thuc, progress=False)
        
        if data.empty:
            print(f"❌ Không thể tải dữ liệu cho mã {ma_ck}")
            return None
            
        print(f"✅ Đã tải thành công {len(data)} ngày dữ liệu")
        return data
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None

def tinh_chi_bao_ky_thuat(df):
    """
    Tính toán các chỉ báo kỹ thuật cơ bản
    
    Args:
        df (pd.DataFrame): Dữ liệu giá chứng khoán
    
    Returns:
        pd.DataFrame: Dữ liệu với các chỉ báo kỹ thuật
    """
    print("🔍 Đang tính toán chỉ báo kỹ thuật...")
    
    df = df.copy()
    
    # SMA (Simple Moving Average)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # EMA (Exponential Moving Average)
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price change
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(5)
    
    print("✅ Đã tính toán xong các chỉ báo kỹ thuật")
    return df

def phan_tich_xu_huong(df):
    """
    Phân tích xu hướng thị trường
    
    Args:
        df (pd.DataFrame): Dữ liệu giá với chỉ báo kỹ thuật
    
    Returns:
        dict: Kết quả phân tích xu hướng
    """
    latest = df.iloc[-1]
    
    # Xu hướng ngắn hạn (5 ngày)
    short_trend = "Tăng" if latest['Close'] > df['Close'].iloc[-6] else "Giảm"
    
    # Xu hướng trung hạn (20 ngày)
    medium_trend = "Tăng" if latest['Close'] > df['SMA_20'] else "Giảm"
    
    # Xu hướng dài hạn (50 ngày)
    long_trend = "Tăng" if latest['Close'] > df['SMA_50'] else "Giảm"
    
    # RSI signal
    if latest['RSI'] > 70:
        rsi_signal = "Quá mua"
    elif latest['RSI'] < 30:
        rsi_signal = "Quá bán"
    else:
        rsi_signal = "Bình thường"
    
    # MACD signal
    macd_signal = "Tích cực" if latest['MACD'] > latest['MACD_Signal'] else "Tiêu cực"
    
    return {
        'xu_huong_ngan_han': short_trend,
        'xu_huong_trung_han': medium_trend,
        'xu_huong_dai_han': long_trend,
        'rsi_tin_hieu': rsi_signal,
        'macd_tin_hieu': macd_signal,
        'gia_hien_tai': latest['Close'],
        'rsi_gia_tri': latest['RSI'],
        'macd_gia_tri': latest['MACD']
    }

def hien_thi_ket_qua_phan_tich(ma_ck, xu_huong):
    """
    Hiển thị kết quả phân tích một cách đẹp mắt
    
    Args:
        ma_ck (str): Mã chứng khoán
        xu_huong (dict): Kết quả phân tích xu hướng
    """
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ PHÂN TÍCH - {ma_ck}")
    print(f"{'='*60}")
    
    print(f"💰 Giá hiện tại: {xu_huong['gia_hien_tai']:,.0f} VND")
    print(f"📈 Xu hướng ngắn hạn: {xu_huong['xu_huong_ngan_han']}")
    print(f"📊 Xu hướng trung hạn: {xu_huong['xu_huong_trung_han']}")
    print(f"📉 Xu hướng dài hạn: {xu_huong['xu_huong_dai_han']}")
    print(f"🔍 RSI (14): {xu_huong['rsi_gia_tri']:.1f} - {xu_huong['rsi_tin_hieu']}")
    print(f"📊 MACD: {xu_huong['macd_gia_tri']:.4f} - {xu_huong['macd_tin_hieu']}")
    
    print(f"{'='*60}\n")

def ve_bieu_do_phan_tich(df, ma_ck):
    """
    Vẽ biểu đồ phân tích đẹp mắt
    
    Args:
        df (pd.DataFrame): Dữ liệu giá với chỉ báo kỹ thuật
        ma_ck (str): Mã chứng khoán
    """
    # Thiết lập font tiếng Việt
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    # Tạo subplot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Biểu đồ giá và SMA
    ax1.plot(df.index, df['Close'], label='Giá đóng cửa', linewidth=2, color='blue')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', linewidth=1, color='orange')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1, color='red')
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2, color='gray', label='Bollinger Bands')
    ax1.set_title(f'Biểu đồ giá {ma_ck}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Giá (VND)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Biểu đồ RSI
    ax2.plot(df.index, df['RSI'], label='RSI', linewidth=2, color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Quá mua (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Quá bán (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Trung tính (50)')
    ax2.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Biểu đồ MACD
    ax3.plot(df.index, df['MACD'], label='MACD', linewidth=2, color='blue')
    ax3.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=2, color='red')
    ax3.bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.6, color=['green' if x >= 0 else 'red' for x in df['MACD_Histogram']])
    ax3.set_title('MACD', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Hàm chính để chạy phân tích chứng khoán
    """
    print("🚀 BẮT ĐẦU PHÂN TÍCH CHỨNG KHOÁN")
    print("="*50)
    
    # Cấu hình
    ma_chung_khoan = "VCB"  # Có thể thay đổi thành mã khác
    ngay_bat_dau = "2023-01-01"
    
    # Tải dữ liệu
    data = tai_du_lieu_chung_khoan(ma_chung_khoan, ngay_bat_dau)
    
    if data is not None:
        # Tính toán chỉ báo kỹ thuật
        data_with_indicators = tinh_chi_bao_ky_thuat(data)
        
        # Phân tích xu hướng
        xu_huong = phan_tich_xu_huong(data_with_indicators)
        
        # Hiển thị kết quả
        hien_thi_ket_qua_phan_tich(ma_chung_khoan, xu_huong)
        
        # Vẽ biểu đồ
        ve_bieu_do_phan_tich(data_with_indicators, ma_chung_khoan)
        
        # Lưu dữ liệu
        data_with_indicators.to_csv(f"data/{ma_chung_khoan}_phan_tich.csv")
        print(f"💾 Đã lưu dữ liệu phân tích vào data/{ma_chung_khoan}_phan_tich.csv")
        
    else:
        print("❌ Không thể thực hiện phân tích do lỗi tải dữ liệu")

if __name__ == "__main__":
    main()