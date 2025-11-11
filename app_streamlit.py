import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import sys
import joblib

warnings.filterwarnings('ignore')

# CẢI TIẾN: Thêm các module xử lý dữ liệu và mô hình một cách nhất quán
from features.feature_engineering import them_chi_bao_ky_thuat, add_technical_features
from ml_model import forecast_with_model

# Thêm path để import gemini client
# Giữ nguyên phần này
sys.path.append(os.path.join(os.path.dirname(__file__), 'gemini'))
try:
    from gemini.gemini_client import predict_stock_price_with_gemini, predict_multi_timeframe_with_gemini, analyze_market_sentiment_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==========================
# CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="Phân Tích Chứng Khoán Thông Minh",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Giữ nguyên CSS của bạn
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4;}
    .success-metric {color: #00C851; font-size: 1.2rem; font-weight: bold;}
    .warning-metric {color: #ff8800; font-size: 1.2rem; font-weight: bold;}
    .danger-metric {color: #ff4444; font-size: 1.2rem; font-weight: bold;}
    .prediction-summary {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;}
    .prediction-number {font-size: 2rem; font-weight: bold; margin: 0.5rem 0;}
    .prediction-detail {font-size: 1.1rem; opacity: 0.9;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Hệ Thống Phân Tích Chứng Khoán với Gemini AI</h1>', unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("🎛️ Bảng Điều Khiển")

# Cấu hình Gemini API (Giữ nguyên)
st.sidebar.subheader("🤖 Cấu Hình Gemini AI")
if not GEMINI_AVAILABLE:
    st.sidebar.error("❌ Gemini AI không khả dụng")
else:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Lấy API key tại: https://makersuite.google.com/app/apikey")
    if api_key:
        os.environ['GEMINI_API_KEY'] = api_key
        st.sidebar.success("✅ API Key đã được cấu hình!")

st.sidebar.subheader("📈 Lựa Chọn Phân Tích")
ma_chung_khoan = st.sidebar.selectbox(
    "Chọn Mã Chứng Khoán",
    ["VCB", "VIC", "VHM", "HPG", "MSN", "CTG", "TCB", "GAS", "VRE", "PLX"],
    index=0
)
ngay_bat_dau = st.sidebar.date_input("Ngày bắt đầu", value=datetime.now() - timedelta(days=365))
ngay_ket_thuc = st.sidebar.date_input("Ngày kết thúc", value=datetime.now())

st.sidebar.subheader("🧠 Dự Báo ML (XGBoost)")
ml_enabled = st.sidebar.checkbox("Bật dự báo ML", value=True)
if ml_enabled:
    ml_horizon = st.sidebar.slider("Số ngày dự báo", min_value=1, max_value=10, value=5)

# ==========================
# CÁC HÀM XỬ LÝ DỮ LIỆU
# ==========================

# File: app_streamlit.py

@st.cache_data(ttl=300)
def tai_va_xu_ly_du_lieu(ma_ck, ngay_bd, ngay_kt):
    """
    Tải và xử lý dữ liệu. Phiên bản này được tối ưu để xử lý các định dạng 
    khác nhau từ yfinance một cách mạnh mẽ nhất.
    """
    try:
        # Tải dữ liệu, thử với hậu tố .VN trước
        ticker_vn = f"{ma_ck}.VN"
        data = yf.download(ticker_vn, start=ngay_bd, end=ngay_kt, progress=False)

        # Nếu không có dữ liệu, thử lại với mã gốc
        if data.empty:
            data = yf.download(ma_ck, start=ngay_bd, end=ngay_kt, progress=False)
        
        # Nếu vẫn không có dữ liệu, dừng lại
        if data.empty:
            st.error(f"Không thể tải được bất kỳ dữ liệu nào cho mã {ma_ck} trong khoảng thời gian đã chọn.")
            return None

        # SỬA LỖI QUAN TRỌNG NHẤT: Xử lý cấu trúc cột đa cấp (MultiIndex)
        # Nếu yfinance trả về cột dạng ('Open', 'VCB.VN'), chúng ta sẽ làm phẳng nó
        if isinstance(data.columns, pd.MultiIndex):
            # Giữ lại cấp độ đầu tiên ('Open', 'Close',...) và loại bỏ cấp độ thứ hai ('VCB.VN')
            data.columns = data.columns.get_level_values(0)

        # Ghi lại tên cột gốc để gỡ lỗi nếu cần
        original_columns = data.columns.tolist()

        # Chuẩn hóa tất cả tên cột thành chữ thường
        data.columns = [str(col).lower() for col in data.columns]
        
        # Logic chuẩn hóa cột 'close' (ưu tiên giá đã điều chỉnh)
        if 'adj close' in data.columns:
            data = data.rename(columns={'adj close': 'close'})
        
        # Kiểm tra cuối cùng, nếu vẫn không có cột 'close', báo lỗi chi tiết
        if 'close' not in data.columns:
            st.error(f"Lỗi Dữ Liệu: Dữ liệu tải về cho {ma_ck} không có cột 'close' hoặc 'adj close'.")
            st.warning("Các cột nhận được là:")
            st.code(original_columns)
            return None

        # Tiếp tục xử lý như bình thường
        data_with_features = them_chi_bao_ky_thuat(data)
        data_with_features = add_technical_features(data_with_features)
        
        return data_with_features.dropna()
        
    except Exception as e:
        st.error(f"Lỗi không xác định trong quá trình tải và xử lý dữ liệu: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def phan_tich_xu_huong(df):
    """
    Phân tích xu hướng từ dữ liệu đã có chỉ báo.
    SỬA LỖI: Sử dụng tên cột chữ thường.
    """
    if df is None or len(df) < 50:
        return {} # Trả về dict rỗng nếu không đủ dữ liệu
    
    latest = df.iloc[-1]
    
    # Hàm helper để lấy giá trị an toàn
    def get_safe_float(series, default=0.0):
        return float(series) if pd.notna(series) else default

    gia_hien_tai = get_safe_float(latest['close'])
    gia_6_ngay_truoc = get_safe_float(df['close'].iloc[-6])
    sma_20 = get_safe_float(latest['sma_20'])
    sma_50 = get_safe_float(latest['sma_50'])
    rsi = get_safe_float(latest['rsi_14'], 50)
    macd = get_safe_float(latest['macd'])
    macd_signal = get_safe_float(latest['macd_signal'])
    
    return {
        "Ngắn hạn": "Tăng" if gia_hien_tai > gia_6_ngay_truoc else "Giảm",
        "Trung hạn": "Tăng" if gia_hien_tai > sma_20 else "Giảm",
        "Dài hạn": "Tăng" if gia_hien_tai > sma_50 else "Giảm",
        "RSI": "Quá mua" if rsi > 70 else "Quá bán" if rsi < 30 else "Bình thường",
        "MACD": "Tích cực" if macd > macd_signal else "Tiêu cực"
    }

@st.cache_data(ttl=900)
def du_bao_da_khung_thoi_gian_gemini(ticker, df, xu_huong):
    """
    Hàm gọi Gemini AI, đảm bảo dữ liệu đầu vào đúng định dạng.
    SỬA LỖI: Sử dụng tên cột chữ thường.
    """
    if not GEMINI_AVAILABLE: return None
    try:
        latest = df.iloc[-1]
        historical_data = {
            'current_price': float(latest['close']),
            'high_52w': float(df['high'].rolling(min(252, len(df))).max().iloc[-1]),
            'low_52w': float(df['low'].rolling(min(252, len(df))).min().iloc[-1]),
            'avg_volume': float(df['volume'].rolling(20).mean().iloc[-1]),
            'rsi': float(latest['rsi_14']),
            'macd': float(latest['macd']),
            'sma_20': float(latest['sma_20']),
            'sma_50': float(latest['sma_50'])
        }
        market_conditions = {
            'short_term_trend': xu_huong.get('Ngắn hạn', 'N/A'),
            'medium_term_trend': xu_huong.get('Trung hạn', 'N/A'),
            'long_term_trend': xu_huong.get('Dài hạn', 'N/A'),
            'rsi_status': xu_huong.get('RSI', 'N/A'),
            'macd_status': xu_huong.get('MACD', 'N/A')
        }
        return predict_multi_timeframe_with_gemini(ticker, historical_data, market_conditions)
    except Exception as e:
        st.error(f"Lỗi khi dự báo đa khung thời gian với Gemini: {str(e)}")
        return None

# ==========================
# GIAO DIỆN CHÍNH
# ==========================
with st.spinner("⏳ Đang tải và xử lý dữ liệu..."):
    data = tai_va_xu_ly_du_lieu(ma_chung_khoan, ngay_bat_dau, ngay_ket_thuc)

if data is not None and not data.empty:
    xu_huong = phan_tich_xu_huong(data)
    
    # ===== THÔNG TIN CƠ BẢN =====
    st.subheader(f"📊 Thông Tin Cơ Bản - {ma_chung_khoan}")
    col1, col2, col3, col4 = st.columns(4)
    # SỬA LỖI: Sử dụng 'close', 'volume', 'high', 'low'
    gia_hien_tai = float(data['close'].iloc[-1])
    gia_truoc = float(data['close'].iloc[-2])
    col1.metric("Giá Hiện Tại", f"{gia_hien_tai:,.0f} VND", f"{gia_hien_tai - gia_truoc:,.0f} VND")
    col2.metric("Khối Lượng GD", f"{float(data['volume'].iloc[-1]):,.0f}")
    col3.metric("Cao Nhất 52T", f"{float(data['high'].rolling(min(252, len(data))).max().iloc[-1]):,.0f} VND")
    col4.metric("Thấp Nhất 52T", f"{float(data['low'].rolling(min(252, len(data))).min().iloc[-1]):,.0f} VND")

    # ===== PHÂN TÍCH XU HƯỚNG =====
    if xu_huong:
        st.subheader("📈 Phân Tích Xu Hướng")
        cols = st.columns(len(xu_huong))
        for i, (ten, gia_tri) in enumerate(xu_huong.items()):
            color_class = "success-metric" if gia_tri in ["Tăng", "Tích cực"] else "danger-metric"
            if gia_tri in ["Quá mua", "Quá bán", "Bình thường"]: color_class = "warning-metric"
            cols[i].markdown(f'<div class="metric-card"><strong>{ten}</strong><br><span class="{color_class}">{gia_tri}</span></div>', unsafe_allow_html=True)
    
    # ===== DỰ BÁO ML (XGBoost) =====
    if ml_enabled:
         st.subheader("🧠 Dự Báo Giá với ML (XGBoost)")

    # Thêm lựa chọn khung thời gian cho người dùng
    timeframe_options = {
        "Ngắn hạn": "short",
        "Dài hạn": "long"
    }
    selected_timeframe_name = st.selectbox(
        "Chọn mô hình huấn luyện:",
        options=list(timeframe_options.keys())
    )
    timeframe_code = timeframe_options[selected_timeframe_name]

    try:
        # SỬA LỖI: Tạo đường dẫn động dựa trên mã cổ phiếu và khung thời gian đã chọn
        model_path = os.path.join(
            os.path.dirname(__file__), 
            "../models", 
            ma_chung_khoan,          # <-- Tự động lấy mã đang được chọn (ví dụ: "VCB")
            f"model_{timeframe_code}.pkl" # <-- Tự động lấy khung thời gian (ví dụ: "model_short.pkl")
        )
        
        # Kiểm tra xem file model có tồn tại không
        if os.path.exists(model_path):
            model, feature_cols = joblib.load(model_path)
            
            with st.spinner(f"🔮 Đang dự báo cho {ma_chung_khoan} bằng mô hình {selected_timeframe_name}..."):
                
                # Kiểm tra xem dữ liệu có đủ các cột cần thiết cho model không
                missing_cols = set(feature_cols) - set(data.columns)
                if not missing_cols:
                    # Nếu đủ, thực hiện dự báo
                    ml_preds = forecast_with_model(model, data, feature_cols, days_ahead=ml_horizon)
                    
                    # Hiển thị kết quả
                    ngay_cuoi = pd.to_datetime(data.index[-1])
                    future_dates = [ngay_cuoi + pd.Timedelta(days=i) for i in range(1, len(ml_preds) + 1)]

                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Giá lịch sử', line=dict(color='#1f77b4')))
                    fig_ml.add_trace(go.Scatter(x=future_dates, y=ml_preds, mode='lines+markers', name='Dự báo ML', line=dict(color='#2ca02c')))
                    fig_ml.update_layout(title=f"Dự Báo ML {ml_horizon} ngày cho {ma_chung_khoan}", yaxis_title="Giá (VND)", height=400)
                    st.plotly_chart(fig_ml, use_container_width=True)
                else:
                    st.warning(f"⚠️ Không thể dự báo ML: Dữ liệu hiện tại thiếu các cột đặc trưng cần thiết cho mô hình - {missing_cols}")
        else:
            st.warning(f"⚠️ Không tìm thấy mô hình cho mã {ma_chung_khoan} (khung {selected_timeframe_name}).")
            st.info(f"Đường dẫn đang tìm kiếm: {model_path}")
            st.info("Vui lòng chạy lại file `train_model.py` để huấn luyện mô hình cho mã này.")
            
    except Exception as e:
        st.error(f"Lỗi khi thực hiện dự báo ML: {e}")

    # ===== DỰ BÁO ĐA KHUNG THỜI GIAN GEMINI =====
# CODE MỚI - CHỈ GỌI API KHI NGƯỜI DÙNG NHẤN NÚT
if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
    
    st.subheader("🤖 Phân Tích Chuyên Sâu với Gemini AI")
    # Thêm một nút bấm để người dùng chủ động kích hoạt
    if st.button("📈 Chạy Phân Tích & Dự Báo AI "):
        
        # Toàn bộ logic gọi API và hiển thị kết quả được chuyển vào BÊN TRONG khối if của nút bấm
        with st.spinner("🤖 Đang liên hệ với chuyên gia AI..."):
            multi_timeframe_prediction = du_bao_da_khung_thoi_gian_gemini(ma_chung_khoan, data, xu_huong)
        
        if multi_timeframe_prediction:
            st.subheader("🕒 Kết Quả Dự Báo Đa Khung Thời Gian")
            
            try:
                def _to_row(pred):
                    if not pred: return None
                    return {
                        "Khung thời gian": pred.timeframe or "",
                        "Giá dự báo": float(pred.predicted_price) if pred.predicted_price is not None else np.nan,
                        "Xu hướng": pred.trend or "Ổn định",
                        "Độ tin cậy (%)": float(pred.confidence) if pred.confidence is not None else 50.0,
                        "Rủi ro": pred.risk_level or "Trung bình",
                        "Lý do": pred.reasoning or "N/A"
                    }
                
                rows = [
                    _to_row(getattr(multi_timeframe_prediction, key, None))
                    for key in ["short_term_3d", "short_term_5d", "short_term_1w", "medium_term_1m", "medium_term_3m", "long_term_6m", "long_term_1y"]
                ]
                rows = [r for r in rows if r is not None]

                if rows:
                    df_multi = pd.DataFrame(rows)
                    st.dataframe(df_multi, use_container_width=True)

                    # Detect likely fallback from Gemini: all predicted prices equal current price
                    try:
                        preds = df_multi['Giá dự báo'].to_numpy(dtype=float)
                        confs = df_multi['Độ tin cậy (%)'].to_numpy(dtype=float)
                        reasons = df_multi['Lý do'].astype(str).to_list()
                        # If predictions are all (nearly) equal to current price OR confidences are all ~50
                        if len(preds) > 0 and (np.allclose(preds, gia_hien_tai, rtol=1e-6) or np.allclose(confs, 50.0, rtol=1e-3)):
                            # Show warning and reasoning to help debugging
                            st.warning('⚠️ Gemini có vẻ trả về dự báo mặc định (một đường thẳng). Thông thường do AI trả về kết quả không phải JSON hoặc lỗi API/Quota.')
                            # Show unique reasons/messages from the AI
                            unique_reasons = list(dict.fromkeys([r for r in reasons if r and r.lower() not in ['n/a', 'none']]))
                            if unique_reasons:
                                st.markdown('**Lý do trả về từ Gemini (tóm tắt):**')
                                for r in unique_reasons:
                                    st.code(r)
                            else:
                                st.info('Không có thông tin lý do chi tiết. Kiểm tra logs/terminal để xem raw response.')
                    except Exception:
                        # non-fatal: nếu có lỗi khi kiểm tra, chỉ bỏ qua
                        pass

                    fig_multi = go.Figure()
                    fig_multi.add_trace(go.Scatter(x=df_multi["Khung thời gian"], y=df_multi["Giá dự báo"], mode="lines+markers", name="Dự báo AI"))
                    # Giả sử biến 'gia_hien_tai' đã được định nghĩa ở phía trên trong code của bạn
                    fig_multi.add_hline(y=gia_hien_tai, line_dash="dash", annotation_text=f"Giá hiện tại: {gia_hien_tai:,.0f}")
                    fig_multi.update_layout(title="Dự báo giá theo khung thời gian", yaxis_title="Giá (VND)", height=400)
                    st.plotly_chart(fig_multi, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Không thể hiển thị dự báo đa khung thời gian: {e}")
    else:
        # Hiển thị thông báo hướng dẫn khi người dùng chưa nhấn nút
        st.info("Nhấn nút ở trên để bắt đầu phân tích và dự báo bằng Gemini AI.")

elif GEMINI_AVAILABLE:
    st.warning("⚠️ Vui lòng nhập Gemini API key ở sidebar để sử dụng tính năng dự báo AI.")

    # ===== BIỂU ĐỒ GIÁ & CHỈ BÁO =====
    st.subheader("📉 Biểu Đồ Giá & Chỉ Báo")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.1)
    
    # SỬA LỖI: Sử dụng tên cột chữ thường
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Giá"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], mode="lines", name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], mode="lines", name="SMA 50"), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['macd'], mode="lines", name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], mode="lines", name="MACD Signal"), row=2, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Không thể tải hoặc xử lý dữ liệu. Vui lòng kiểm tra lại mã chứng khoán và khoảng thời gian.")