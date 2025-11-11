import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import  VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

def add_technical_features(df):
    df = df.copy()
    df['Close_lag_1'] = df['close'].shift(1)
    df['Close_lag_3'] = df['close'].shift(3)
    df['Close_lag_5'] = df['close'].shift(5)
    df['Close_lag_10'] = df['close'].shift(10)
    
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['vol_mean_20'] = df['volume'].rolling(20).mean()
    df['vol_std_20'] = df['volume'].rolling(20).std()
    
    return df.dropna()
def them_chi_bao_ky_thuat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame giá chứng khoán
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu giá (OHLCV)
    
    Returns:
        pd.DataFrame: DataFrame với các chỉ báo kỹ thuật đã được thêm
    """
    print("Dang them chi bao ky thuat...")
    
    df = df.copy()
    so_cot_ban_dau = len(df.columns)
    
    # Chuẩn hóa tên cột - xử lý MultiIndex columns từ yfinance
    # Chỉ báo xu hướng (Trend Indicators)
    # SMA - Simple Moving Average
    df['sma_5'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
    df['sma_10'] = SMAIndicator(close=df['close'], window=10).sma_indicator()
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    
    # EMA - Exponential Moving Average
    df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
    
    # MACD - Moving Average Convergence Divergence
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # ADX - Average Directional Index
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    
    # Chỉ báo momentum (Momentum Indicators)
    # RSI - Relative Strength Index
    df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi_21'] = RSIIndicator(close=df['close'], window=21).rsi()
    
    # ROC - Rate of Change
    df['roc_10'] = ROCIndicator(close=df['close'], window=10).roc()
    df['roc_20'] = ROCIndicator(close=df['close'], window=20).roc()
    
    # Williams %R
    df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
    
    # Chỉ báo biến động (Volatility Indicators)
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Average True Range
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # Chỉ báo khối lượng (Volume Indicators)
    # Volume SMA
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    
    # VWAP - Volume Weighted Average Price
    df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Tính toán lợi nhuận và biến động
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    
    # Volatility
    df['vol_5d'] = df['return_1d'].rolling(5).std()
    df['vol_20d'] = df['return_1d'].rolling(20).std()
    
    # High-Low ratio
    df['hl_ratio'] = df['high'] / df['low']
    
    # Price position trong ngày
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Gap
    df['gap'] = df['close'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)
    
    # Loại bỏ các dòng có giá trị NaN
    df = df.dropna()
    
    print(f"Da them {len(df.columns) - so_cot_ban_dau} chi bao ky thuat")
    return df

def gop_tinh_nang_bai_viet(price_df: pd.DataFrame, su_kien_bai_viet: list) -> pd.DataFrame:
    """
    Gộp tính năng từ bài viết tin tức vào DataFrame giá
    
    Args:
        price_df (pd.DataFrame): DataFrame chứa dữ liệu giá
        su_kien_bai_viet (list): Danh sách các sự kiện từ bài viết với keys: 
                                ['ngay', 'ma_ck', 'cam_xuc', 'tac_dong']
    
    Returns:
        pd.DataFrame: DataFrame đã được gộp với tính năng từ bài viết
    """
    print("📰 Đang gộp tính năng từ bài viết...")
    
    df = price_df.copy()
    df = df.sort_index()
    
    # Chuyển đổi danh sách sự kiện thành DataFrame
    ev_df = pd.DataFrame(su_kien_bai_viet)
    
    if ev_df.empty:
        df['cam_xuc_1d'] = 0.0
        df['so_su_kien_1d'] = 0
        return df
    
    # Chuẩn hóa tên cột
    if 'date' in ev_df.columns:
        ev_df = ev_df.rename(columns={'date': 'ngay'})
    if 'sentiment' in ev_df.columns:
        ev_df = ev_df.rename(columns={'sentiment': 'cam_xuc'})
    if 'impact' in ev_df.columns:
        ev_df = ev_df.rename(columns={'impact': 'tac_dong'})
    
    # Chuyển đổi ngày thành datetime
    ev_df['ngay'] = pd.to_datetime(ev_df['ngay']).dt.normalize()
    
    # Nhóm theo ngày và tính toán các thống kê
    agg_stats = ev_df.groupby('ngay').agg({
        'cam_xuc': ['mean', 'std', 'count'],
        'tac_dong': 'count'
    }).round(4)
    
    # Làm phẳng tên cột
    agg_stats.columns = ['cam_xuc_trung_binh', 'cam_xuc_do_lech_chuan', 'so_bai_viet', 'so_su_kien']
    
    # Gộp với DataFrame giá
    df = df.join(agg_stats, how='left')
    
    # Điền các giá trị NaN
    df['cam_xuc_trung_binh'] = df['cam_xuc_trung_binh'].fillna(0.0)
    df['cam_xuc_do_lech_chuan'] = df['cam_xuc_do_lech_chuan'].fillna(0.0)
    df['so_bai_viet'] = df['so_bai_viet'].fillna(0)
    df['so_su_kien'] = df['so_su_kien'].fillna(0)
    
    # Tính toán các tính năng bổ sung
    df['cam_xuc_tich_cuc'] = (df['cam_xuc_trung_binh'] > 0.1).astype(int)
    df['cam_xuc_tieu_cuc'] = (df['cam_xuc_trung_binh'] < -0.1).astype(int)
    df['tin_tuc_nhieu'] = (df['so_bai_viet'] > df['so_bai_viet'].rolling(20).mean()).astype(int)
    
    print("✅ Đã gộp thành công tính năng từ bài viết")
    return df

def tao_tinh_nang_nang_cao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các tính năng nâng cao từ dữ liệu giá và chỉ báo kỹ thuật
    
    Args:
        df (pd.DataFrame): DataFrame đã có chỉ báo kỹ thuật
    
    Returns:
        pd.DataFrame: DataFrame với các tính năng nâng cao
    """
    print("🚀 Đang tạo tính năng nâng cao...")
    
    df = df.copy()
    
    # Tính năng tương tác giữa các chỉ báo
    df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Tín hiệu mua/bán từ RSI
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # Tín hiệu từ Bollinger Bands
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
    df['bb_breakout_up'] = (df['close'] > df['bb_upper']).astype(int)
    df['bb_breakout_down'] = (df['close'] < df['bb_lower']).astype(int)
    
    # Momentum features
    df['momentum_strong'] = ((df['rsi_14'] > 50) & (df['macd'] > df['macd_signal'])).astype(int)
    df['momentum_weak'] = ((df['rsi_14'] < 50) & (df['macd'] < df['macd_signal'])).astype(int)
    
    # Volume features
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    df['volume_dry'] = (df['volume_ratio'] < 0.5).astype(int)
    
    # Price action features
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    
    # Volatility features
    df['high_vol'] = (df['vol_20d'] > df['vol_20d'].rolling(50).quantile(0.8)).astype(int)
    df['low_vol'] = (df['vol_20d'] < df['vol_20d'].rolling(50).quantile(0.2)).astype(int)
    
    # Trend strength
    df['trend_strength'] = abs(df['adx'])
    df['strong_trend'] = (df['trend_strength'] > 25).astype(int)
    
    # Gap features
    df['gap_up'] = (df['gap'] > 0).astype(int)
    df['gap_down'] = (df['gap'] < 0).astype(int)
    df['big_gap'] = (abs(df['gap_pct']) > 0.03).astype(int)
    
    print("✅ Đã tạo thành công các tính năng nâng cao")
    return df

def chuan_hoa_du_lieu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu để chuẩn bị cho machine learning
    
    Args:
        df (pd.DataFrame): DataFrame với các tính năng
    
    Returns:
        pd.DataFrame: DataFrame đã được chuẩn hóa
    """
    print("🔧 Đang chuẩn hóa dữ liệu...")
    
    df = df.copy()
    
    # Loại bỏ các cột không cần thiết cho ML
    columns_to_drop = ['open', 'high', 'low', 'close', 'volume']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns)
    
    # Loại bỏ các dòng có giá trị vô hạn
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Chuẩn hóa các cột số
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Min-Max scaling cho các cột có giá trị lớn
    large_value_columns = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 
                          'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'vwap']
    
    for col in large_value_columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    print(f"✅ Đã chuẩn hóa {len(numeric_columns)} cột dữ liệu")
    return df

# Backward compatibility
add_technical_indicators = them_chi_bao_ky_thuat
merge_article_features = gop_tinh_nang_bai_viet