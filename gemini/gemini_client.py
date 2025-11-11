import os
import json
import time
import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional
from google.api_core import exceptions as google_exceptions

# Cấu hình Gemini API động theo biến môi trường
_CURRENT_GENAI_KEY = None

def _ensure_gemini_configured() -> None:
    """Đảm bảo google.generativeai luôn dùng API key mới nhất từ biến môi trường.
    Gọi hàm này trước mỗi lần gọi model để tránh tình trạng cấu hình từ lần import đầu.
    """
    global _CURRENT_GENAI_KEY
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key.strip().lower() in {'', 'your-api-key-here', 'none'}:
        # Không ném Exception ở đây để phía UI có thể hiển thị fallback hợp lý
        _CURRENT_GENAI_KEY = None
        return
    if api_key != _CURRENT_GENAI_KEY:
        genai.configure(api_key=api_key)
        _CURRENT_GENAI_KEY = api_key

def _select_supported_model(preferred_models: list) -> Optional[str]:
    """Chọn model khả dụng có hỗ trợ generateContent.
    Trả về tên model hoặc None nếu không thể xác định.
    """
    try:
        available = list(genai.list_models())
    except Exception:
        # Nếu không list được models (mạng/API), cứ thử dùng ưu tiên đầu
        return preferred_models[0] if preferred_models else None

    # Lập chỉ mục theo tên phục vụ tra cứu nhanh
    name_to_model = {m.name: m for m in available if hasattr(m, 'supported_generation_methods')}

    # Ưu tiên đúng tên trong danh sách ưu tiên
    for name in preferred_models:
        m = name_to_model.get(name)
        if m and 'generateContent' in getattr(m, 'supported_generation_methods', []):
            return name

    # Nếu không khớp chính xác, cố gắng tìm model có tên chứa chuỗi ưu tiên
    for name in preferred_models:
        for m in available:
            if name in m.name and 'generateContent' in getattr(m, 'supported_generation_methods', []):
                return m.name

    # Fallback: trả về bất kỳ model nào hỗ trợ generateContent
    for m in available:
        if 'generateContent' in getattr(m, 'supported_generation_methods', []):
            return m.name
    return None

def _resolve_and_get_model(initial_model: Optional[str]) -> str:
    """Trả về model hợp lệ, ưu tiên theo danh sách và model đầu vào nếu có."""
    preferred = [
        # Danh sách phổ biến, cập nhật theo SDK mới
        'gemini-1.5-flash-8b',
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro',
        'gemini-1.0-pro',
    ]
    if initial_model:
        preferred = [initial_model] + [m for m in preferred if m != initial_model]
    chosen = _select_supported_model(preferred)
    return chosen or (initial_model or 'gemini-1.5-flash')

def _summarize_error_message(error_text: str) -> str:
    """Rút gọn thông báo lỗi để hiển thị gọn trong UI tiếng Việt."""
    text = str(error_text)
    lower = text.lower()
    if 'api key' in lower or 'api_key_invalid' in lower:
        return 'API key không hợp lệ. Vui lòng kiểm tra và nhập lại.'
    if 'quota' in lower or 'rate limit' in lower or '429' in lower:
        # Thử tách retry seconds nếu có
        retry = None
        import re
        m = re.search(r'retry(?:_delay)?\s*\{\s*seconds:\s*(\d+)', text)
        if not m:
            m = re.search(r'retry in\s*(\d+(?:\.\d+)?)s', lower)
        if m:
            retry = m.group(1)
        if retry:
            return f'Vượt giới hạn quota. Vui lòng thử lại sau ~{retry} giây hoặc nâng gói/bật billing.'
        return 'Vượt giới hạn quota. Vui lòng thử lại sau hoặc nâng gói/bật billing.'
    if 'not found' in lower or 'not supported' in lower:
        return 'Model không khả dụng với API hiện tại. Đã tự chuyển model khác.'
    return 'Có lỗi khi gọi Gemini AI. Vui lòng thử lại sau.'

class NewsEvent(BaseModel):
    ticker: Optional[str] = None
    event_type: Optional[str] = None
    impact: Optional[str] = None
    sentiment: Optional[float] = 0.0
    summary: Optional[str] = None

class StockPrediction(BaseModel):
    predicted_price: Optional[float] = None
    confidence: Optional[float] = None
    trend: Optional[str] = None
    reasoning: Optional[str] = None
    risk_level: Optional[str] = None
    timeframe: Optional[str] = None  # Thêm khung thời gian

class MultiTimeframePrediction(BaseModel):
    short_term_3d: Optional[StockPrediction] = None
    short_term_5d: Optional[StockPrediction] = None
    short_term_1w: Optional[StockPrediction] = None
    medium_term_1m: Optional[StockPrediction] = None
    medium_term_3m: Optional[StockPrediction] = None
    long_term_6m: Optional[StockPrediction] = None
    long_term_1y: Optional[StockPrediction] = None




PROMPT_TEMPLATE = '''
You are a financial news analyzer. Given the article title and article body, return a JSON object with the following keys:
- ticker: main company ticker if present (string or null)
- event_type: one of [earnings, merger, guidance, product, regulation, other]
- impact: one of [high, medium, low]
- sentiment: float between -1.0 and 1.0 (negative: bad for stock, positive: good)
- summary: short 1-2 sentence summary


Return ONLY valid JSON.
Article Title: {title}
Article Body: {body}
'''




def safe_parse_json(text: str):
    # Try to extract JSON object from text
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # some LLMs add extra text. try to find first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None


def _try_extract_json_candidates(text: str):
    """Try several heuristics to extract a JSON object from a noisy LLM response.
    Returns the parsed object or None.
    """
    # 1) direct json
    parsed = safe_parse_json(text)
    if parsed is not None:
        return parsed

    # 2) try to find all {...} blocks and parse each
    import re
    candidates = re.findall(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL) if hasattr(re, 'findall') else []
    # fallback simpler regex if recursive pattern not available
    if not candidates:
        candidates = re.findall(r"\{[^}]+\}", text, re.DOTALL)

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    # 3) try to extract between first '{' and last '}' (already in safe_parse_json), try again with trimming
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        cand = text[start:end+1]
        # try to fix common issues: replace single quotes, remove trailing commas
        cand_fixed = cand.replace("'", '"')
        cand_fixed = re.sub(r",\s*}\s*$", '}', cand_fixed)
        try:
            return json.loads(cand_fixed)
        except Exception:
            return None

    return None




def analyze_article_with_gemini(title: str, body: str, model: str = 'gemini-1.5-flash') -> NewsEvent:
    prompt = PROMPT_TEMPLATE.format(title=title, body=body)
    try:
        _ensure_gemini_configured()
        use_model = _resolve_and_get_model(model)
        model_instance = genai.GenerativeModel(use_model)
        response = model_instance.generate_content(prompt)
        text = response.text
    except Exception as e:
        # Nếu lỗi do model không tồn tại/không hỗ trợ, thử fallback lần nữa
        err = str(e)
        if 'not found' in err.lower() or 'not supported' in err.lower():
            try:
                use_model = _resolve_and_get_model(None)
                model_instance = genai.GenerativeModel(use_model)
                response = model_instance.generate_content(prompt)
                text = response.text
            except Exception as e2:
                raise RuntimeError(_summarize_error_message(str(e2)))
        else:
            raise RuntimeError(_summarize_error_message(err))

    parsed = safe_parse_json(text)
    if parsed is None:
        return NewsEvent(summary=text[:200])
    
    if 'sentiment' in parsed:
        try:
            parsed['sentiment'] = float(parsed['sentiment'])
        except Exception:
            parsed['sentiment'] = 0.0
    return NewsEvent(**parsed)

def predict_stock_price_with_gemini(ticker: str, historical_data: dict, market_conditions: dict, model: str = 'gemini-1.5-flash') -> StockPrediction:
    """
    Dự báo giá cổ phiếu sử dụng Gemini AI
    """
    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính AI. Hãy phân tích và dự báo giá cổ phiếu {ticker} dựa trên dữ liệu sau:

    Dữ liệu lịch sử:
    - Giá hiện tại: {historical_data.get('current_price', 'N/A')} VND
    - Giá cao nhất 52 tuần: {historical_data.get('high_52w', 'N/A')} VND
    - Giá thấp nhất 52 tuần: {historical_data.get('low_52w', 'N/A')} VND
    - Khối lượng giao dịch trung bình: {historical_data.get('avg_volume', 'N/A')}
    - RSI: {historical_data.get('rsi', 'N/A')}
    - MACD: {historical_data.get('macd', 'N/A')}
    - SMA 20: {historical_data.get('sma_20', 'N/A')} VND
    - SMA 50: {historical_data.get('sma_50', 'N/A')} VND

    Điều kiện thị trường:
    - Xu hướng ngắn hạn: {market_conditions.get('short_term_trend', 'N/A')}
    - Xu hướng trung hạn: {market_conditions.get('medium_term_trend', 'N/A')}
    - Xu hướng dài hạn: {market_conditions.get('long_term_trend', 'N/A')}
    - Tình trạng RSI: {market_conditions.get('rsi_status', 'N/A')}
    - Tình trạng MACD: {market_conditions.get('macd_status', 'N/A')}

    Hãy phân tích và đưa ra dự báo cụ thể:
    1. Tính toán giá dự báo dựa trên xu hướng và chỉ báo kỹ thuật
    2. Đánh giá mức độ tin cậy của dự báo
    3. Xác định xu hướng và mức độ rủi ro
    4. Giải thích lý do phân tích chi tiết

    Trả về JSON với các thông tin sau:
    - predicted_price: giá dự báo cụ thể (float, tính bằng VND)
    - confidence: độ tin cậy từ 0-100 (float)
    - trend: xu hướng ["Tăng mạnh", "Tăng nhẹ", "Ổn định", "Giảm nhẹ", "Giảm mạnh"]
    - reasoning: lý do phân tích chi tiết, bao gồm mức tăng/giảm cụ thể (string)
    - risk_level: mức độ rủi ro ["Thấp", "Trung bình", "Cao"]

    Ví dụ reasoning: "Dự báo giá tăng 5.2% (+3,200 VND) do xu hướng trung hạn tích cực và RSI đang ở vùng quá bán..."

    Chỉ trả về JSON hợp lệ, không có text khác.
    """
    
    try:
        _ensure_gemini_configured()
        use_model = _resolve_and_get_model(model)
        model_instance = genai.GenerativeModel(use_model)
        response = model_instance.generate_content(prompt)
        text = response.text
    except Exception as e:
        return StockPrediction(
            predicted_price=historical_data.get('current_price', 0),
            confidence=50.0,
            trend="Ổn định",
            reasoning=_summarize_error_message(str(e)),
            risk_level="Trung bình"
        )

    parsed = safe_parse_json(text)
    if parsed is None:
        return StockPrediction(
            predicted_price=historical_data.get('current_price', 0),
            confidence=50.0,
            trend="Ổn định",
            reasoning="Không thể phân tích dữ liệu",
            risk_level="Trung bình"
        )
    
    return StockPrediction(**parsed)
# File: gemini/gemini_client.py

def predict_multi_timeframe_with_gemini(ticker: str, historical_data: dict, market_conditions: dict, model: str = 'gemini-1.5-flash') -> MultiTimeframePrediction:
    """
    Dự báo giá cổ phiếu cho nhiều khung thời gian sử dụng Gemini AI.
    *** PHIÊN BẢN NÂNG CẤP: TỰ ĐỘNG THỬ LẠI KHI GẶP LỖI QUOTA ***
    """
    # ... (Phần prompt giữ nguyên, không thay đổi) ...
    prompt = f"""
    ...
    """
    
    # --- LOGIC TỰ ĐỘNG THỬ LẠI ---
    max_retries = 3  # Thử lại tối đa 3 lần
    base_wait_time = 20 # Thời gian chờ ban đầu (giây)

    for attempt in range(max_retries):
        try:
            _ensure_gemini_configured()
            use_model = _resolve_and_get_model(model)
            model_instance = genai.GenerativeModel(use_model)
            
            print(f"Attempt {attempt + 1}: Calling Gemini API...") # In ra để bạn theo dõi
            response = model_instance.generate_content(prompt)
            text = response.text
            
            # Nếu thành công, phân tích và trả về kết quả
            # Try robust JSON extraction (support noisy LLM outputs)
            parsed = _try_extract_json_candidates(text)
            if parsed:
                # parsed expected to be a dict of timeframe -> prediction-dicts
                prediction_objects = {k: StockPrediction(**v) for k, v in parsed.items()}
                return MultiTimeframePrediction(**prediction_objects)
            else:
                # Log raw response for debugging and raise to trigger retry/fallback
                print("=== Gemini raw response (non-JSON) ===")
                print(text[:2000])
                print("=== end raw response ===")
                raise ValueError("AI response is not valid JSON")

        # Bắt lỗi cụ thể của Google Cloud khi hết quota
        except google_exceptions.ResourceExhausted as e:
            print(f"Quota exceeded on attempt {attempt + 1}. Waiting to retry...")
            # Nếu là lần thử cuối cùng, trả về lỗi
            if attempt == max_retries - 1:
                error_reason = _summarize_error_message(str(e))
                break # Thoát vòng lặp để trả về lỗi mặc định
            
            # Chờ đợi trước khi thử lại
            time.sleep(base_wait_time)
            
        except Exception as e:
            # Bắt các lỗi khác (API key sai, mạng,...) và trả về ngay
            print(f"An unexpected error occurred: {e}")
            error_reason = _summarize_error_message(str(e))
            break # Thoát vòng lặp

    # --- PHẦN DỰ PHÒNG ---
    # Chỉ chạy khi tất cả các lần thử lại đều thất bại
    current_price = historical_data.get('current_price', 0)
    # Nếu error_reason chưa được định nghĩa, gán một lỗi chung
    if 'error_reason' not in locals():
        error_reason = "All retry attempts failed."
        
    default_prediction = StockPrediction(predicted_price=current_price, confidence=50, trend="Ổn định", reasoning=error_reason, risk_level="Trung bình")
    return MultiTimeframePrediction(
        short_term_3d=default_prediction.model_copy(update={'timeframe': '3 ngày'}),
        short_term_5d=default_prediction.model_copy(update={'timeframe': '5 ngày'}),
        short_term_1w=default_prediction.model_copy(update={'timeframe': '1 tuần'}),
        medium_term_1m=default_prediction.model_copy(update={'timeframe': '1 tháng'}),
        medium_term_3m=default_prediction.model_copy(update={'timeframe': '3 tháng'}),
        long_term_6m=default_prediction.model_copy(update={'timeframe': '6 tháng'}),
        long_term_1y=default_prediction.model_copy(update={'timeframe': '1 năm'})
    )
    
    # Thêm .copy() để tránh lỗi Pydantic
    # Chuyển đổi dict của dicts thành các đối tượng StockPrediction
    prediction_objects = {k: StockPrediction(**v) for k, v in parsed.items()}
    return MultiTimeframePrediction(**prediction_objects)

def analyze_market_sentiment_with_gemini(news_articles: list, model: str = 'gemini-1.5-flash') -> dict:
    """
    Phân tích sentiment thị trường từ tin tức
    """
    if not news_articles:
        return {
            'overall_sentiment': 0.0,
            'sentiment_score': 'Trung tính',
            'market_outlook': 'Ổn định',
            'key_insights': ['Không có tin tức để phân tích']
        }
    
    articles_text = "\n".join([f"Tiêu đề: {article.get('title', '')}\nNội dung: {article.get('content', '')}" for article in news_articles[:5]])
    
    prompt = f"""
    Phân tích sentiment thị trường chứng khoán Việt Nam từ các tin tức sau:

    {articles_text}

    Hãy trả về JSON với:
    - overall_sentiment: điểm sentiment từ -1.0 đến 1.0 (float)
    - sentiment_score: mô tả ["Rất tích cực", "Tích cực", "Trung tính", "Tiêu cực", "Rất tiêu cực"]
    - market_outlook: triển vọng thị trường ["Tăng mạnh", "Tăng", "Ổn định", "Giảm", "Giảm mạnh"]
    - key_insights: danh sách các insight chính (array of strings)

    Chỉ trả về JSON hợp lệ.
    """
    
    try:
        _ensure_gemini_configured()
        use_model = _resolve_and_get_model(model)
        model_instance = genai.GenerativeModel(use_model)
        response = model_instance.generate_content(prompt)
        text = response.text
    except Exception as e:
        return {
            'overall_sentiment': 0.0,
            'sentiment_score': 'Trung tính',
            'market_outlook': 'Ổn định',
            'key_insights': [f'Lỗi phân tích: {_summarize_error_message(str(e))}']
        }

    parsed = safe_parse_json(text)
    if parsed is None:
        return {
            'overall_sentiment': 0.0,
            'sentiment_score': 'Trung tính',
            'market_outlook': 'Ổn định',
            'key_insights': ['Không thể phân tích sentiment']
        }
    
    return parsed




if __name__ == '__main__':
    sample_title = 'Company X reports strong earnings beat for Q2'
    sample_body = 'Company X reported revenue growth of 30% and beat EPS estimates...'
    ev = analyze_article_with_gemini(sample_title, sample_body)
    print(ev.model_dump_json())