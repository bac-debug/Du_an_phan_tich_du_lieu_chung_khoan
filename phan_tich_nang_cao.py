"""
Module phân tích nâng cao cho chứng khoán
Bao gồm các tính năng: Dự đoán giá, Phân tích sentiment, Cảnh báo rủi ro
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DuDoanGia:
    """
    Lớp dự đoán giá chứng khoán sử dụng nhiều thuật toán ML
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ket_qua_danh_gia = {}
    
    def khoi_tao_models(self):
        """
        Khởi tạo các model machine learning
        """
        print("🤖 Đang khởi tạo các model ML...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Khởi tạo scaler cho từng model
        for name in self.models.keys():
            self.scalers[name] = StandardScaler()
        
        print("✅ Đã khởi tạo thành công các model")
    
    def chuan_bi_du_lieu(self, df, cot_muc_tieu='return_1d', so_ngay_truoc=5):
        """
        Chuẩn bị dữ liệu cho việc training
        
        Args:
            df (pd.DataFrame): DataFrame với các chỉ báo kỹ thuật
            cot_muc_tieu (str): Cột mục tiêu để dự đoán
            so_ngay_truoc (int): Số ngày trước đó để làm features
        """
        print(f"🔧 Đang chuẩn bị dữ liệu cho việc dự đoán {cot_muc_tieu}...")
        
        # Chọn các cột feature
        cot_features = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi_14', 'rsi_21', 'roc_10', 'roc_20',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'volume_ratio', 'return_1d', 'return_5d',
            'vol_5d', 'vol_20d', 'hl_ratio', 'price_position'
        ]
        
        # Chỉ lấy các cột có trong DataFrame
        cot_features = [col for col in cot_features if col in df.columns]
        
        # Tạo lag features
        df_features = df[cot_features].copy()
        for i in range(1, so_ngay_truoc + 1):
            for col in cot_features:
                df_features[f'{col}_lag_{i}'] = df_features[col].shift(i)
        
        # Loại bỏ NaN
        df_features = df_features.dropna()
        
        # Tạo target
        target = df[cot_muc_tieu].shift(-1)  # Dự đoán ngày mai
        
        # Căn chỉnh index
        df_features = df_features.loc[df_features.index.intersection(target.index)]
        target = target.loc[target.index.intersection(df_features.index)]
        
        self.X = df_features
        self.y = target
        self.feature_names = df_features.columns.tolist()
        
        print(f"✅ Đã chuẩn bị {len(self.X)} mẫu với {len(self.feature_names)} features")
        
        return df_features, target
    
    def huan_luyen_models(self, test_size=0.2):
        """
        Huấn luyện tất cả các models
        """
        print("🎯 Bắt đầu huấn luyện các models...")
        
        if not self.models:
            self.khoi_tao_models()
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, shuffle=False
        )
        
        for name, model in self.models.items():
            print(f"   📊 Đang huấn luyện {name}...")
            
            try:
                # Chuẩn hóa dữ liệu
                X_train_scaled = self.scalers[name].fit_transform(X_train)
                X_test_scaled = self.scalers[name].transform(X_test)
                
                # Huấn luyện model
                model.fit(X_train_scaled, y_train)
                
                # Dự đoán
                y_pred = model.predict(X_test_scaled)
                
                # Đánh giá
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.ket_qua_danh_gia[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'RMSE': np.sqrt(mse)
                }
                
                # Feature importance (nếu có)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        self.feature_names, model.feature_importances_
                    ))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(
                        self.feature_names, np.abs(model.coef_)
                    ))
                
                print(f"      ✅ {name}: R² = {r2:.4f}, RMSE = {np.sqrt(mse):.6f}")
                
            except Exception as e:
                print(f"      ❌ Lỗi khi huấn luyện {name}: {str(e)}")
        
        print("🎉 Hoàn thành huấn luyện tất cả models")
    
    def chon_model_tot_nhat(self):
        """
        Chọn model có hiệu suất tốt nhất
        """
        if not self.ket_qua_danh_gia:
            print("❌ Chưa có kết quả đánh giá")
            return None
        
        # Chọn model có R² cao nhất
        model_tot_nhat = max(
            self.ket_qua_danh_gia.keys(),
            key=lambda x: self.ket_qua_danh_gia[x]['R2']
        )
        
        print(f"🏆 Model tốt nhất: {model_tot_nhat}")
        print(f"   R² = {self.ket_qua_danh_gia[model_tot_nhat]['R2']:.4f}")
        print(f"   RMSE = {self.ket_qua_danh_gia[model_tot_nhat]['RMSE']:.6f}")
        
        return model_tot_nhat
    
    def du_doan_gia(self, du_lieu_moi, ten_model=None):
        """
        Dự đoán giá cho dữ liệu mới
        
        Args:
            du_lieu_moi (pd.DataFrame): Dữ liệu mới để dự đoán
            ten_model (str): Tên model để sử dụng (nếu None thì dùng model tốt nhất)
        """
        if ten_model is None:
            ten_model = self.chon_model_tot_nhat()
        
        if ten_model not in self.models:
            print(f"❌ Model {ten_model} không tồn tại")
            return None
        
        # Chuẩn bị dữ liệu
        X_new = du_lieu_moi[self.feature_names]
        X_new_scaled = self.scalers[ten_model].transform(X_new)
        
        # Dự đoán
        du_doan = self.models[ten_model].predict(X_new_scaled)
        
        return du_doan
    
    def hien_thi_ket_qua_danh_gia(self):
        """
        Hiển thị kết quả đánh giá các models
        """
        if not self.ket_qua_danh_gia:
            print("❌ Chưa có kết quả đánh giá")
            return
        
        print("\n" + "="*80)
        print("📊 KẾT QUẢ ĐÁNH GIÁ CÁC MODELS")
        print("="*80)
        
        # Tạo DataFrame kết quả
        df_ket_qua = pd.DataFrame(self.ket_qua_danh_gia).T
        df_ket_qua = df_ket_qua.round(6)
        
        # Sắp xếp theo R²
        df_ket_qua = df_ket_qua.sort_values('R2', ascending=False)
        
        print(df_ket_qua)
        
        # Hiển thị model tốt nhất
        model_tot_nhat = df_ket_qua.index[0]
        print(f"\n🏆 Model tốt nhất: {model_tot_nhat}")
        print(f"   R² = {df_ket_qua.loc[model_tot_nhat, 'R2']:.4f}")
        print(f"   RMSE = {df_ket_qua.loc[model_tot_nhat, 'RMSE']:.6f}")
        
        return df_ket_qua

class PhanTichSentiment:
    """
    Lớp phân tích sentiment từ tin tức và dữ liệu thị trường
    """
    
    def __init__(self):
        self.tu_vung_tich_cuc = {
            'tăng', 'tăng trưởng', 'tích cực', 'tốt', 'mạnh', 'cải thiện',
            'lợi nhuận', 'thành công', 'breakthrough', 'vượt', 'vượt trội',
            'tích cực', 'khả quan', 'hy vọng', 'triển vọng', 'tăng trưởng'
        }
        
        self.tu_vung_tieu_cuc = {
            'giảm', 'suy giảm', 'tiêu cực', 'xấu', 'yếu', 'xấu đi',
            'thua lỗ', 'thất bại', 'khủng hoảng', 'sụt giảm', 'khó khăn',
            'tiêu cực', 'bi quan', 'lo ngại', 'rủi ro', 'suy thoái'
        }
    
    def phan_tich_sentiment_van_ban(self, van_ban):
        """
        Phân tích sentiment của văn bản tiếng Việt
        
        Args:
            van_ban (str): Văn bản cần phân tích
        
        Returns:
            dict: Kết quả phân tích sentiment
        """
        if not van_ban or pd.isna(van_ban):
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'Trung tính'}
        
        van_ban = str(van_ban).lower()
        tu_vung = van_ban.split()
        
        diem_tich_cuc = sum(1 for tu in tu_vung if tu in self.tu_vung_tich_cuc)
        diem_tieu_cuc = sum(1 for tu in tu_vung if tu in self.tu_vung_tieu_cuc)
        
        tong_tu = len(tu_vung)
        if tong_tu == 0:
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'Trung tính'}
        
        # Tính điểm sentiment (-1 đến 1)
        sentiment_score = (diem_tich_cuc - diem_tieu_cuc) / tong_tu
        
        # Tính confidence
        confidence = abs(diem_tich_cuc - diem_tieu_cuc) / tong_tu
        
        # Xác định label
        if sentiment_score > 0.1:
            label = 'Tích cực'
        elif sentiment_score < -0.1:
            label = 'Tiêu cực'
        else:
            label = 'Trung tính'
        
        return {
            'sentiment': sentiment_score,
            'confidence': confidence,
            'label': label,
            'diem_tich_cuc': diem_tich_cuc,
            'diem_tieu_cuc': diem_tieu_cuc
        }
    
    def phan_tich_sentiment_thi_truong(self, df):
        """
        Phân tích sentiment dựa trên các chỉ báo thị trường
        
        Args:
            df (pd.DataFrame): DataFrame với các chỉ báo kỹ thuật
        
        Returns:
            pd.DataFrame: DataFrame với sentiment scores
        """
        print("📊 Đang phân tích sentiment thị trường...")
        
        df = df.copy()
        
        # Sentiment từ RSI
        df['sentiment_rsi'] = np.where(
            df['rsi_14'] < 30, 0.8,  # Quá bán -> tích cực
            np.where(df['rsi_14'] > 70, -0.8, 0)  # Quá mua -> tiêu cực
        )
        
        # Sentiment từ MACD
        df['sentiment_macd'] = np.where(
            df['macd'] > df['macd_signal'], 0.6,  # MACD > Signal -> tích cực
            -0.6  # MACD < Signal -> tiêu cực
        )
        
        # Sentiment từ Bollinger Bands
        df['sentiment_bb'] = np.where(
            df['close'] < df['bb_lower'], 0.7,  # Giá dưới BB -> tích cực
            np.where(df['close'] > df['bb_upper'], -0.7, 0)  # Giá trên BB -> tiêu cực
        )
        
        # Sentiment từ Volume
        df['sentiment_volume'] = np.where(
            df['volume_ratio'] > 1.5, 0.3,  # Volume cao -> tích cực
            np.where(df['volume_ratio'] < 0.5, -0.3, 0)  # Volume thấp -> tiêu cực
        )
        
        # Sentiment từ Price Change
        df['sentiment_price'] = np.tanh(df['return_1d'] * 10)  # Normalize price change
        
        # Tổng hợp sentiment
        cot_sentiment = ['sentiment_rsi', 'sentiment_macd', 'sentiment_bb', 
                        'sentiment_volume', 'sentiment_price']
        
        df['sentiment_tong_hop'] = df[cot_sentiment].mean(axis=1)
        
        # Sentiment label
        df['sentiment_label'] = np.where(
            df['sentiment_tong_hop'] > 0.2, 'Tích cực',
            np.where(df['sentiment_tong_hop'] < -0.2, 'Tiêu cực', 'Trung tính')
        )
        
        print("✅ Đã phân tích sentiment thị trường")
        return df

class CanhBaoRuiRo:
    """
    Lớp cảnh báo rủi ro cho giao dịch chứng khoán
    """
    
    def __init__(self):
        self.nguong_canh_bao = {
            'rsi_qua_mua': 80,
            'rsi_qua_ban': 20,
            'volatility_cao': 0.05,  # 5% daily volatility
            'volume_spike': 3.0,  # 3x average volume
            'price_drop': -0.05,  # 5% daily drop
            'bb_breakout': 2.0,  # 2 standard deviations
            'drawdown': -0.15  # 15% drawdown
        }
    
    def kiem_tra_rui_ro(self, df):
        """
        Kiểm tra các rủi ro trong dữ liệu
        
        Args:
            df (pd.DataFrame): DataFrame với các chỉ báo kỹ thuật
        
        Returns:
            pd.DataFrame: DataFrame với các cảnh báo rủi ro
        """
        print("⚠️ Đang kiểm tra rủi ro...")
        
        df = df.copy()
        
        # Cảnh báo RSI
        df['canh_bao_rsi_qua_mua'] = df['rsi_14'] > self.nguong_canh_bao['rsi_qua_mua']
        df['canh_bao_rsi_qua_ban'] = df['rsi_14'] < self.nguong_canh_bao['rsi_qua_ban']
        
        # Cảnh báo volatility
        df['canh_bao_volatility_cao'] = df['vol_20d'] > self.nguong_canh_bao['volatility_cao']
        
        # Cảnh báo volume spike
        df['canh_bao_volume_spike'] = df['volume_ratio'] > self.nguong_canh_bao['volume_spike']
        
        # Cảnh báo giá giảm mạnh
        df['canh_bao_price_drop'] = df['return_1d'] < self.nguong_canh_bao['price_drop']
        
        # Cảnh báo Bollinger Bands breakout
        df['canh_bao_bb_breakout'] = (
            (df['close'] > df['bb_upper']) | 
            (df['close'] < df['bb_lower'])
        )
        
        # Tính drawdown
        df['cum_return'] = (1 + df['return_1d']).cumprod()
        df['peak'] = df['cum_return'].expanding().max()
        df['drawdown'] = (df['cum_return'] - df['peak']) / df['peak']
        
        # Cảnh báo drawdown
        df['canh_bao_drawdown'] = df['drawdown'] < self.nguong_canh_bao['drawdown']
        
        # Tổng số cảnh báo
        cot_canh_bao = [col for col in df.columns if col.startswith('canh_bao_')]
        df['tong_canh_bao'] = df[cot_canh_bao].sum(axis=1)
        
        # Mức độ rủi ro
        df['muc_do_rui_ro'] = np.where(
            df['tong_canh_bao'] >= 4, 'Cao',
            np.where(df['tong_canh_bao'] >= 2, 'Trung bình', 'Thấp')
        )
        
        print("✅ Đã hoàn thành kiểm tra rủi ro")
        return df
    
    def tao_bao_cao_canh_bao(self, df):
        """
        Tạo báo cáo cảnh báo chi tiết
        
        Args:
            df (pd.DataFrame): DataFrame với các cảnh báo rủi ro
        
        Returns:
            dict: Báo cáo cảnh báo
        """
        if 'tong_canh_bao' not in df.columns:
            df = self.kiem_tra_rui_ro(df)
        
        # Thống kê cảnh báo
        tong_ngay = len(df)
        ngay_rui_ro_cao = len(df[df['muc_do_rui_ro'] == 'Cao'])
        ngay_rui_ro_trung_binh = len(df[df['muc_do_rui_ro'] == 'Trung bình'])
        ngay_rui_ro_thap = len(df[df['muc_do_rui_ro'] == 'Thấp'])
        
        # Cảnh báo gần đây
        canh_bao_gan_day = df.tail(5)
        
        # Tổng hợp cảnh báo
        tong_canh_bao = {
            'rsi_qua_mua': canh_bao_gan_day['canh_bao_rsi_qua_mua'].sum(),
            'rsi_qua_ban': canh_bao_gan_day['canh_bao_rsi_qua_ban'].sum(),
            'volatility_cao': canh_bao_gan_day['canh_bao_volatility_cao'].sum(),
            'volume_spike': canh_bao_gan_day['canh_bao_volume_spike'].sum(),
            'price_drop': canh_bao_gan_day['canh_bao_price_drop'].sum(),
            'bb_breakout': canh_bao_gan_day['canh_bao_bb_breakout'].sum(),
            'drawdown': canh_bao_gan_day['canh_bao_drawdown'].sum()
        }
        
        bao_cao = {
            'tong_quan': {
                'tong_ngay': tong_ngay,
                'ngay_rui_ro_cao': ngay_rui_ro_cao,
                'ngay_rui_ro_trung_binh': ngay_rui_ro_trung_binh,
                'ngay_rui_ro_thap': ngay_rui_ro_thap,
                'ty_le_rui_ro_cao': ngay_rui_ro_cao / tong_ngay * 100
            },
            'canh_bao_gan_day': tong_canh_bao,
            'muc_do_rui_ro_hien_tai': df['muc_do_rui_ro'].iloc[-1],
            'tong_canh_bao_hien_tai': df['tong_canh_bao'].iloc[-1]
        }
        
        return bao_cao
    
    def hien_thi_bao_cao_canh_bao(self, df):
        """
        Hiển thị báo cáo cảnh báo rủi ro
        """
        bao_cao = self.tao_bao_cao_canh_bao(df)
        
        print("\n" + "="*60)
        print("⚠️ BÁO CÁO CẢNH BÁO RỦI RO")
        print("="*60)
        
        print(f"📊 Tổng quan:")
        print(f"   • Tổng số ngày phân tích: {bao_cao['tong_quan']['tong_ngay']}")
        print(f"   • Ngày rủi ro cao: {bao_cao['tong_quan']['ngay_rui_ro_cao']} ({bao_cao['tong_quan']['ty_le_rui_ro_cao']:.1f}%)")
        print(f"   • Ngày rủi ro trung bình: {bao_cao['tong_quan']['ngay_rui_ro_trung_binh']}")
        print(f"   • Ngày rủi ro thấp: {bao_cao['tong_quan']['ngay_rui_ro_thap']}")
        
        print(f"\n🚨 Mức độ rủi ro hiện tại: {bao_cao['muc_do_rui_ro_hien_tai']}")
        print(f"📈 Tổng số cảnh báo hiện tại: {bao_cao['tong_canh_bao_hien_tai']}")
        
        print(f"\n⚠️ Cảnh báo gần đây (5 ngày):")
        for loai_canh_bao, so_lan in bao_cao['canh_bao_gan_day'].items():
            if so_lan > 0:
                print(f"   • {loai_canh_bao.replace('_', ' ').title()}: {so_lan} lần")
        
        print("="*60)

# Hàm tiện ích
def chay_phan_tich_nang_cao(df, ma_chung_khoan="VCB"):
    """
    Chạy phân tích nâng cao hoàn chỉnh
    
    Args:
        df (pd.DataFrame): DataFrame với dữ liệu giá và chỉ báo kỹ thuật
        ma_chung_khoan (str): Mã chứng khoán
    """
    print(f"🚀 BẮT ĐẦU PHÂN TÍCH NÂNG CAO - {ma_chung_khoan}")
    print("="*70)
    
    # 1. Dự đoán giá
    print("\n1️⃣ DỰ ĐOÁN GIÁ")
    print("-" * 30)
    du_doan = DuDoanGia()
    du_doan.chuan_bi_du_lieu(df)
    du_doan.huan_luyen_models()
    du_doan.hien_thi_ket_qua_danh_gia()
    
    # 2. Phân tích sentiment
    print("\n2️⃣ PHÂN TÍCH SENTIMENT")
    print("-" * 30)
    sentiment = PhanTichSentiment()
    df_with_sentiment = sentiment.phan_tich_sentiment_thi_truong(df)
    
    # 3. Cảnh báo rủi ro
    print("\n3️⃣ CẢNH BÁO RỦI RO")
    print("-" * 30)
    canh_bao = CanhBaoRuiRo()
    df_with_risk = canh_bao.kiem_tra_rui_ro(df_with_sentiment)
    canh_bao.hien_thi_bao_cao_canh_bao(df_with_risk)
    
    print("\n🎉 HOÀN THÀNH PHÂN TÍCH NÂNG CAO")
    print("="*70)
    
    return {
        'du_doan': du_doan,
        'sentiment': sentiment,
        'canh_bao': canh_bao,
        'du_lieu_da_xu_ly': df_with_risk
    }

if __name__ == "__main__":
    # Test với dữ liệu mẫu
    print("🧪 Đang test phân tích nâng cao...")
    
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Tạo dữ liệu giá giả lập
    price_data = []
    price = 100000
    for i in range(100):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        price_data.append({
            'close': price,
            'high': price * (1 + abs(np.random.normal(0, 0.01))),
            'low': price * (1 - abs(np.random.normal(0, 0.01))),
            'volume': np.random.randint(1000000, 5000000)
        })
    
    df_test = pd.DataFrame(price_data, index=dates)
    
    # Thêm các chỉ báo kỹ thuật cơ bản
    df_test['return_1d'] = df_test['close'].pct_change()
    df_test['sma_20'] = df_test['close'].rolling(20).mean()
    df_test['rsi_14'] = 50 + np.random.normal(0, 15, 100)  # RSI giả lập
    df_test['macd'] = np.random.normal(0, 0.001, 100)
    df_test['macd_signal'] = np.random.normal(0, 0.001, 100)
    df_test['bb_upper'] = df_test['close'] * 1.02
    df_test['bb_lower'] = df_test['close'] * 0.98
    df_test['volume_ratio'] = np.random.uniform(0.5, 2.0, 100)
    df_test['vol_20d'] = np.random.uniform(0.01, 0.05, 100)
    
    # Chạy phân tích
    ket_qua = chay_phan_tich_nang_cao(df_test, "TEST")
