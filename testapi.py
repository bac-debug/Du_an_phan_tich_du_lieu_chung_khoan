import google.generativeai as genai
import os

# === Nhập API key của bạn vào đây ===
# Tốt hơn là dùng biến môi trường để bảo mật
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = "-NHLWHJK4S4_k" # Thay key của bạn vào đây

try:
    genai.configure(api_key=GOOGLE_API_KEY)

    print("✅ Đã cấu hình thành công. Đang lấy danh sách models...")

    # Lấy danh sách tất cả các model hỗ trợ generateContent
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

except Exception as e:
    print("❌ Vẫn có lỗi xảy ra:")
    print(str(e))