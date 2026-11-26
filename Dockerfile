FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดไฟล์โมเดล buffalo_l ตรงๆ แล้วลบไฟล์เสริมที่ไม่ใช้ออก (ไม่กิน RAM ตอน Build)
RUN python -c "import urllib.request, zipfile, os; \
    os.makedirs('/root/.insightface/models/buffalo_l', exist_ok=True); \
    urllib.request.urlretrieve('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', '/tmp/b.zip'); \
    zipfile.ZipFile('/tmp/b.zip').extractall('/root/.insightface/models/buffalo_l'); \
    os.remove('/tmp/b.zip')" \
    && rm -f /root/.insightface/models/buffalo_l/genderage.onnx \
    /root/.insightface/models/buffalo_l/1k3d68.onnx \
    /root/.insightface/models/buffalo_l/2d106det.onnx

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
