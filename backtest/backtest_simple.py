import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Thiết lập font tiếng Việt cho matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans']

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'

class PhanTichBacktest:
    """
    Lớp phân tích backtest cho các chiến lược giao dịch
    """
    
    def __init__(self, df, ten_chien_luoc="Chiến lược cơ bản"):
        self.df = df.copy()
        self.ten_chien_luoc = ten_chien_luoc
        self.ket_qua = None
    
    def chien_luoc_co_ban(self, cot_du_doan='pred', nguong_mua=0.0, nguong_ban=0.0):
        """
        Chiến lược cơ bản: Mua khi dự đoán > nguong_mua, bán khi dự đoán < nguong_ban
        
        Args:
            cot_du_doan (str): Tên cột chứa dự đoán
            nguong_mua (float): Ngưỡng để mua
            nguong_ban (float): Ngưỡng để bán
        """
        print(f"🎯 Áp dụng chiến lược: {self.ten_chien_luoc}")
        
        df = self.df.copy()
        
        # Tạo tín hiệu mua/bán
        df['tin_hieu'] = 0
        df.loc[df[cot_du_doan] > nguong_mua, 'tin_hieu'] = 1  # Mua
        df.loc[df[cot_du_doan] < nguong_ban, 'tin_hieu'] = -1  # Bán
        
        # Tính lợi nhuận chiến lược (sử dụng tín hiệu của ngày trước)
        df['loi_nhuan_chien_luoc'] = df['tin_hieu'].shift(1) * df['return_1d']
        
        # Tính lợi nhuận tích lũy
        df['loi_nhuan_tich_luy_chien_luoc'] = (1 + df['loi_nhuan_chien_luoc'].fillna(0)).cumprod()
        df['loi_nhuan_tich_luy_mua_giu'] = (1 + df['return_1d'].fillna(0)).cumprod()
        
        self.ket_qua = df
        return df
    
    def chien_luoc_rsi(self, rsi_thap=30, rsi_cao=70):
        """
        Chiến lược dựa trên RSI: Mua khi RSI < rsi_thap, bán khi RSI > rsi_cao
        
        Args:
            rsi_thap (float): Ngưỡng RSI để mua
            rsi_cao (float): Ngưỡng RSI để bán
        """
        print(f"🎯 Áp dụng chiến lược RSI: Mua < {rsi_thap}, Bán > {rsi_cao}")
        
        df = self.df.copy()
        
        # Tạo tín hiệu dựa trên RSI
        df['tin_hieu'] = 0
        df.loc[df['rsi_14'] < rsi_thap, 'tin_hieu'] = 1  # Mua khi quá bán
        df.loc[df['rsi_14'] > rsi_cao, 'tin_hieu'] = -1  # Bán khi quá mua
        
        # Tính lợi nhuận
        df['loi_nhuan_chien_luoc'] = df['tin_hieu'].shift(1) * df['return_1d']
        df['loi_nhuan_tich_luy_chien_luoc'] = (1 + df['loi_nhuan_chien_luoc'].fillna(0)).cumprod()
        df['loi_nhuan_tich_luy_mua_giu'] = (1 + df['return_1d'].fillna(0)).cumprod()
        
        self.ket_qua = df
        return df
    
    def chien_luoc_macd(self):
        """
        Chiến lược dựa trên MACD: Mua khi MACD > Signal, bán khi MACD < Signal
        """
        print("🎯 Áp dụng chiến lược MACD")
        
        df = self.df.copy()
        
        # Tạo tín hiệu dựa trên MACD
        df['tin_hieu'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'tin_hieu'] = 1  # Mua
        df.loc[df['macd'] < df['macd_signal'], 'tin_hieu'] = -1  # Bán
        
        # Tính lợi nhuận
        df['loi_nhuan_chien_luoc'] = df['tin_hieu'].shift(1) * df['return_1d']
        df['loi_nhuan_tich_luy_chien_luoc'] = (1 + df['loi_nhuan_chien_luoc'].fillna(0)).cumprod()
        df['loi_nhuan_tich_luy_mua_giu'] = (1 + df['return_1d'].fillna(0)).cumprod()
        
        self.ket_qua = df
        return df
    
def chien_luoc_bollinger_bands(self):
        """
        Chiến lược Bollinger Bands: Mua khi giá chạm dải dưới, bán khi giá chạm dải trên
        """
        print("🎯 Áp dụng chiến lược Bollinger Bands")
        
        df = self.df.copy()
        
        # Tạo tín hiệu dựa trên Bollinger Bands
        df['tin_hieu'] = 0
        df.loc[df['close'] <= df['bb_lower'], 'tin_hieu'] = 1  # Mua khi giá chạm dải dưới
        df.loc[df['close'] >= df['bb_upper'], 'tin_hieu'] = -1  # Bán khi giá chạm dải trên
        
        # Tính lợi nhuận
        df['loi_nhuan_chien_luoc'] = df['tin_hieu'].shift(1) * df['return_1d']
        df['loi_nhuan_tich_luy_chien_luoc'] = (1 + df['loi_nhuan_chien_luoc'].fillna(0)).cumprod()
        df['loi_nhuan_tich_luy_mua_giu'] = (1 + df['return_1d'].fillna(0)).cumprod()
        
        self.ket_qua = df
        # SỬA LỖI: Di chuyển 'return df' ra ngoài.
        # Hoặc tốt hơn là xóa nó đi vì hàm này sửa đổi self.ket_qua, không cần trả về.
        return df # Giữ lại để tương thích nếu có code khác gọi

def tinh_chi_so_hieu_qua(self):
        """
        Tính toán các chỉ số hiệu quả của chiến lược
        """
        if self.ket_qua is None:
            print("❌ Chưa có kết quả backtest")
            return None
        
        df = self.ket_qua
        
        # Lợi nhuận tổng
        loi_nhuan_tong_chien_luoc = df['loi_nhuan_tich_luy_chien_luoc'].iloc[-1] - 1
        loi_nhuan_tong_mua_giu = df['loi_nhuan_tich_luy_mua_giu'].iloc[-1] - 1
        
        # Lợi nhuận trung bình hàng năm
        so_ngay = len(df)
        loi_nhuan_nam_chien_luoc = (1 + loi_nhuan_tong_chien_luoc) ** (252 / so_ngay) - 1
        loi_nhuan_nam_mua_giu = (1 + loi_nhuan_tong_mua_giu) ** (252 / so_ngay) - 1
        
        # Volatility
        vol_chien_luoc = df['loi_nhuan_chien_luoc'].std() * np.sqrt(252)
        vol_mua_giu = df['return_1d'].std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_chien_luoc = loi_nhuan_nam_chien_luoc / vol_chien_luoc if vol_chien_luoc > 0 else 0
        sharpe_mua_giu = loi_nhuan_nam_mua_giu / vol_mua_giu if vol_mua_giu > 0 else 0
        
        # Maximum Drawdown
        dd_chien_luoc = self._tinh_max_drawdown(df['loi_nhuan_tich_luy_chien_luoc'])
        dd_mua_giu = self._tinh_max_drawdown(df['loi_nhuan_tich_luy_mua_giu'])
        
        # Win Rate
        win_rate = (df['loi_nhuan_chien_luoc'] > 0).mean()
        
        chi_so = {
            'loi_nhuan_tong_chien_luoc': loi_nhuan_tong_chien_luoc,
            'loi_nhuan_tong_mua_giu': loi_nhuan_tong_mua_giu,
            'loi_nhuan_nam_chien_luoc': loi_nhuan_nam_chien_luoc,
            'loi_nhuan_nam_mua_giu': loi_nhuan_nam_mua_giu,
            'vol_chien_luoc': vol_chien_luoc,
            'vol_mua_giu': vol_mua_giu,
            'sharpe_chien_luoc': sharpe_chien_luoc,
            'sharpe_mua_giu': sharpe_mua_giu,
            'max_dd_chien_luoc': dd_chien_luoc,
            'max_dd_mua_giu': dd_mua_giu,
            'win_rate': win_rate
        }
        
        return chi_so
    
    def _tinh_max_drawdown(self, cum_returns):
        """Tính maximum drawdown"""
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()
    
    def hien_thi_ket_qua(self):
        """
        Hiển thị kết quả backtest một cách đẹp mắt
        """
        if self.ket_qua is None:
            print("❌ Chưa có kết quả backtest")
            return
        
        chi_so = self.tinh_chi_so_hieu_qua()
        
        print(f"\n{'='*60}")
        print(f"📊 KẾT QUẢ BACKTEST - {self.ten_chien_luoc}")
        print(f"{'='*60}")
        
        print(f"💰 Lợi nhuận tổng:")
        print(f"   • Chiến lược: {chi_so['loi_nhuan_tong_chien_luoc']:,.2%}")
        print(f"   • Mua & Giữ:  {chi_so['loi_nhuan_tong_mua_giu']:,.2%}")
        
        print(f"\n📈 Lợi nhuận hàng năm:")
        print(f"   • Chiến lược: {chi_so['loi_nhuan_nam_chien_luoc']:,.2%}")
        print(f"   • Mua & Giữ:  {chi_so['loi_nhuan_nam_mua_giu']:,.2%}")
        
        print(f"\n📊 Độ biến động:")
        print(f"   • Chiến lược: {chi_so['vol_chien_luoc']:,.2%}")
        print(f"   • Mua & Giữ:  {chi_so['vol_mua_giu']:,.2%}")
        
        print(f"\n🎯 Sharpe Ratio:")
        print(f"   • Chiến lược: {chi_so['sharpe_chien_luoc']:.3f}")
        print(f"   • Mua & Giữ:  {chi_so['sharpe_mua_giu']:.3f}")
        
        print(f"\n📉 Maximum Drawdown:")
        print(f"   • Chiến lược: {chi_so['max_dd_chien_luoc']:,.2%}")
        print(f"   • Mua & Giữ:  {chi_so['max_dd_mua_giu']:,.2%}")
        
        print(f"\n🎲 Win Rate: {chi_so['win_rate']:,.2%}")
        
        # So sánh hiệu quả
        if chi_so['loi_nhuan_nam_chien_luoc'] > chi_so['loi_nhuan_nam_mua_giu']:
            print(f"\n✅ Chiến lược hiệu quả hơn Mua & Giữ")
        else:
            print(f"\n❌ Chiến lược kém hiệu quả hơn Mua & Giữ")
        
        print(f"{'='*60}\n")
    
    def ve_bieu_do_ket_qua(self):
        """
        Vẽ biểu đồ kết quả backtest
        """
        if self.ket_qua is None:
            print("❌ Chưa có kết quả backtest")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Biểu đồ lợi nhuận tích lũy
        ax1.plot(self.ket_qua.index, self.ket_qua['loi_nhuan_tich_luy_chien_luoc'], 
                label='Chiến lược', linewidth=2, color='blue')
        ax1.plot(self.ket_qua.index, self.ket_qua['loi_nhuan_tich_luy_mua_giu'], 
                label='Mua & Giữ', linewidth=2, color='red')
        ax1.set_title(f'Lợi nhuận tích lũy - {self.ten_chien_luoc}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Lợi nhuận tích lũy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Biểu đồ tín hiệu giao dịch
        ax2.plot(self.ket_qua.index, self.ket_qua['close'], label='Giá đóng cửa', linewidth=1, color='black')
        
        # Đánh dấu điểm mua (màu xanh)
        mua = self.ket_qua[self.ket_qua['tin_hieu'] == 1]
        if not mua.empty:
            ax2.scatter(mua.index, mua['close'], color='green', marker='^', s=50, label='Tín hiệu mua')
        
        # Đánh dấu điểm bán (màu đỏ)
        ban = self.ket_qua[self.ket_qua['tin_hieu'] == -1]
        if not ban.empty:
            ax2.scatter(ban.index, ban['close'], color='red', marker='v', s=50, label='Tín hiệu bán')
        
        ax2.set_title('Tín hiệu giao dịch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Giá (VND)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def so_sanh_nhieu_chien_luoc(df, danh_sach_chien_luoc):
    """
    So sánh nhiều chiến lược với nhau
    
    Args:
        df (pd.DataFrame): Dữ liệu giá
        danh_sach_chien_luoc (list): Danh sách các tuple (tên, hàm chiến lược, tham số)
    """
    print("🔄 BẮT ĐẦU SO SÁNH NHIỀU CHIẾN LƯỢC")
    print("="*60)
    
    ket_qua_so_sanh = {}
    
    for ten, ham_chien_luoc, tham_so in danh_sach_chien_luoc:
        print(f"\n📊 Đang test chiến lược: {ten}")
        
        # Tạo backtest instance
        backtest = PhanTichBacktest(df, ten)
        
        # Áp dụng chiến lược
        if tham_so:
            ham_chien_luoc(backtest, **tham_so)
        else:
            ham_chien_luoc(backtest)
        
        # Tính toán chỉ số
        chi_so = backtest.tinh_chi_so_hieu_qua()
        ket_qua_so_sanh[ten] = chi_so
        
        # Hiển thị kết quả
        backtest.hien_thi_ket_qua()
    
    # Tạo bảng so sánh
    bang_so_sanh = pd.DataFrame(ket_qua_so_sanh).T
    bang_so_sanh = bang_so_sanh.round(4)
    
    print("\n📋 BẢNG SO SÁNH CÁC CHIẾN LƯỢC")
    print("="*80)
    print(bang_so_sanh[['loi_nhuan_nam_chien_luoc', 'vol_chien_luoc', 'sharpe_chien_luoc', 'max_dd_chien_luoc']])
    
    return ket_qua_so_sanh

# Backward compatibility
def simple_strategy(df, pred_col='pred'):
    """Hàm cũ để tương thích ngược"""
    backtest = PhanTichBacktest(df, "Chiến lược cơ bản")
    return backtest.chien_luoc_co_ban(pred_col)

if __name__ == '__main__':
    # Tải dữ liệu
    try:
df = pd.read_csv(DATA_DIR / 'VCB_prices.csv', parse_dates=[0], index_col=0)
        from src.features.feature_engineering import them_chi_bao_ky_thuat
        
        # Thêm chỉ báo kỹ thuật
        df = them_chi_bao_ky_thuat(df)
        
        # Tạo dự đoán giả lập dựa trên RSI
df['pred'] = np.where(df['rsi_14'] < 30, 0.01, -0.005)
        
        print("🚀 BẮT ĐẦU PHÂN TÍCH BACKTEST")
        print("="*50)
        
        # So sánh nhiều chiến lược
        danh_sach_chien_luoc = [
            ("Chiến lược RSI", PhanTichBacktest.chien_luoc_rsi, {'rsi_thap': 30, 'rsi_cao': 70}),
            ("Chiến lược MACD", PhanTichBacktest.chien_luoc_macd, None),
            ("Chiến lược Bollinger Bands", PhanTichBacktest.chien_luoc_bollinger_bands, None),
            ("Chiến lược dự đoán", PhanTichBacktest.chien_luoc_co_ban, {'cot_du_doan': 'pred', 'nguong_mua': 0.005, 'nguong_ban': -0.005})
        ]
        
        ket_qua = so_sanh_nhieu_chien_luoc(df, danh_sach_chien_luoc)
        
        # Vẽ biểu đồ cho chiến lược tốt nhất
        chi_strategy_tot_nhat = max(ket_qua.keys(), key=lambda x: ket_qua[x]['sharpe_chien_luoc'])
        print(f"\n🏆 Chiến lược tốt nhất: {chi_strategy_tot_nhat}")
        
        # Test lại chiến lược tốt nhất và vẽ biểu đồ
        backtest_tot_nhat = PhanTichBacktest(df, chi_strategy_tot_nhat)
        if chi_strategy_tot_nhat == "Chiến lược RSI":
            backtest_tot_nhat.chien_luoc_rsi()
        elif chi_strategy_tot_nhat == "Chiến lược MACD":
            backtest_tot_nhat.chien_luoc_macd()
        elif chi_strategy_tot_nhat == "Chiến lược Bollinger Bands":
            backtest_tot_nhat.chien_luoc_bollinger_bands()
        else:
            backtest_tot_nhat.chien_luoc_co_ban()
        
        backtest_tot_nhat.ve_bieu_do_ket_qua()
        
    except FileNotFoundError:
        print("❌ Không tìm thấy file dữ liệu. Vui lòng chạy script tải dữ liệu trước.")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")