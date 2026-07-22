# Hệ thống Đánh giá Rủi ro Tín dụng — Streamlit App

## Cấu trúc thư mục
```
credit-risk-app/
├── app.py                     # Streamlit app
├── requirements.txt           # thư viện cần cài
├── credit_risk_pipeline.pkl   # ⚠️ BẠN CẦN TỰ THÊM FILE NÀY (từ Cell 11 của notebook)
└── README.md
```

## Trước khi chạy: thêm file model

`app.py` cần file `credit_risk_pipeline.pkl` nằm **cùng thư mục**. File này được tạo ở Cell 11
trong notebook gốc (`joblib.dump(...)`). Copy file `.pkl` đó vào thư mục này trước khi
chạy local hoặc push lên GitHub.

## Chạy thử ở local

```bash
cd credit-risk-app
pip install -r requirements.txt
streamlit run app.py
```

App sẽ mở ở `http://localhost:8501`.

## Deploy lên Streamlit Community Cloud

1. Đẩy toàn bộ thư mục này (kèm file `.pkl`) lên 1 repo GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: credit risk streamlit app"
   git remote add origin https://github.com/<username>/credit-risk-app.git
   git branch -M main
   git push -u origin main
   ```
   Nếu `.pkl` > 100MB, dùng Git LFS:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes credit_risk_pipeline.pkl
   git commit -m "Track pkl with LFS"
   git push
   ```

2. Vào **https://share.streamlit.io** → đăng nhập GitHub → **New app**.
3. Chọn repo, branch `main`, main file path `app.py` → **Deploy**.
4. Sau 2–5 phút, app sẽ có URL dạng `https://<tên-app>.streamlit.app`.

## Cập nhật sau này
Chỉ cần `git push` code mới lên nhánh `main`, Streamlit Cloud tự động redeploy.
