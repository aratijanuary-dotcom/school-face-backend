FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# โหลดโมเดล buffalo_l แล้วลบไฟล์เสริม (genderage, 1k3d68, 2d106det) ทิ้งเพื่อลดขนาด Image
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])" \
    && rm -f /root/.insightface/models/buffalo_l/genderage.onnx \
    /root/.insightface/models/buffalo_l/1k3d68.onnx \
    /root/.insightface/models/buffalo_l/2d106det.onnx

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
