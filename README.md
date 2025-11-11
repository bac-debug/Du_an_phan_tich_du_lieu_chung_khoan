<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
 🎓 Faculty of Information Technology (DaiNam University)
    </a>
</h2>
<h2 align="center">
    Hệ Thống Phân Tích và Dự Báo Chứng Khoán
Tích Hợp
AI Tạo Sinh (Gemini) và Học Máy (XGBoost)
</h2>
<div align="center">
    <p align="center">
        <img src="aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="fitdnu_logo (3).png" alt="AIoTLab Logo" width="180"/>
        <img src="dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

# 📈 Hệ Thống Phân Tích và Dự Báo Chứng Khoán (AI + ML)

## 📖 1. Giới thiệu hệ thống
Đây là một ứng dụng web Phân tích Kỹ thuật và Dự báo Chứng khoán, được xây dựng bằng Streamlit và Python. Hệ thống kết hợp cả Học máy (ML) truyền thống và AI tạo sinh (Generative AI) để cung cấp cái nhìn đa chiều cho nhà đầu tư.

- **Người dùng (Nhà đầu tư):** Có thể xem biểu đồ giá, các chỉ báo kỹ thuật, nhận dự báo giá ngắn hạn từ mô hình XGBoost và nhận các phân tích chuyên sâu, đa khung thời gian từ Google Gemini AI.
- **Quy trình Huấn luyện:** Một kịch bản (script) offline được dùng để huấn luyện các mô hình XGBoost cho từng mã cổ phiếu và lưu lại.
- **Giao diện Web:** Ứng dụng Streamlit tải các mô hình đã huấn luyện, đồng thời kết nối trực tiếp đến các API (yfinance, Gemini) để cung cấp dữ liệu và phân tích thời gian thực.

Cấu trúc chính:

- **`app_streamlit.py`**: Giao diện web chính cho người dùng.
- **`train_model.py`**: Kịch bản offline để huấn luyện mô hình ML.
- **`gemini_client.py`**: Client xử lý tất cả logic gọi và phân tích API Gemini.
- **`ml_model.py`**: Chứa logic huấn luyện (XGBoost) và dự báo.
- **`feature_engineering.py`**: Mô-đun tạo các đặc trưng/chỉ báo kỹ thuật.

## 🔧 2. Các công nghệ được sử dụng

- **🐍 Python 3.9+**
- **🌐 Streamlit** (Dựng giao diện web)
- **🧠 Google Gemini API** (Phân tích & Dự báo AI)
- **📈 XGBoost** (Huấn luyện & Dự báo ML)
- **📊 Pandas** & **Numpy** (Xử lý dữ liệu)
- **💹 Plotly** (Vẽ biểu đồ tương tác)
- **🏦 yfinance** (Tải dữ liệu chứng khoán)
- **🛠️ Scikit-learn** & **Joblib** (Hỗ trợ ML & Lưu trữ mô hình)
- **🖥️ VS Code** (Khuyến khích)

## 🚀 3. Một số hình ảnh hệ thống

<p align="center">
    <em>Giao diện chính - Hiển thị biểu đồ giá, chỉ báo kỹ thuật và xu hướng</em><br/>
    <img width="1401" height="842" alt="Main UI" src="[ĐƯỜNG_DẪN_ĐẾN_ẢNH_CỦA_BẠN]" />
</p>
<p align="center">
    <em>Giao diện dự báo ML (XGBoost) hiển thị trên biểu đồ</em><br/>
    <img width="1401" height="842" alt="ML Forecast" src="[ĐƯỜNG_DẪN_ĐẾN_ẢNH_CỦA_BẠN]" />
</p>

<p align="center">
    <em>Kết quả phân tích & dự báo đa khung thời gian từ Gemini AI</em><br/>
    <img width="1387" height="819" alt="Gemini Analysis" src="[ĐƯỜNG_DẪN_ĐẾN_ẢNH_CỦA_BẠN]" />
</p>

---

## ⚙️ 4. Các bước cài đặt

### 4.1. Yêu cầu hệ thống

- Cài đặt Python 3.9 trở lên (kiểm tra bằng lệnh `python --version`).
- Cài đặt Git để clone repository.
- Cài đặt pip để quản lý thư viện (thường đi kèm Python).
- (Khuyến khích) Cài đặt VS Code hoặc PyCharm để dễ quản lý project.


### 4.2. Cài đặt thư viện

1.  Clone repository về máy:
    ```bash
    git clone [ĐƯỜNG_DẪN_REPO_CỦA_BẠN]
    cd Du_an_phan_tich_chung_khoan
    ```

2.  Cài đặt tất cả các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
    *(Lưu ý: Bạn cần tạo tệp `requirements.txt` bằng lệnh `pip freeze > requirements.txt`)*

### 4.3. Bước 1: Huấn luyện mô hình (Offline)

- Chạy file `train_model.py` để huấn luyện các mô hình XGBoost.
- Các mô hình sẽ được lưu vào thư mục `/models/`.
- (Bạn chỉ cần chạy bước này một lần, hoặc mỗi khi muốn cập nhật mô hình).

### 4.4. Bước 2: Chạy Ứng dụng Web

1.  Mở Terminal (hoặc Command Prompt) và di chuyển đến thư mục gốc của dự án.
2.  Gõ lệnh sau và nhấn Enter:
    ```bash
    streamlit run app_streamlit.py
    ```
3.  Mở trình duyệt và truy cập vào địa chỉ (thường là `http://localhost:8501`).
4.  Nhập API Key của Gemini ở thanh bên và bắt đầu sử dụng.

## 📝 5. Liên hệ

- **Khoa:** Công nghệ thông tin - Trường Đại học Đại Nam
- **Lớp:** CNTT 16-04
- **Tôi:** Nguyễn Văn Bắc
- **Email:** nguyenbacdz04@gmail.com

---
*✍️ README này được thiết kế bởi Bac Nguyen*

    
