import os
import glob
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://xxngxkpsrkstnahtmghr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh4bmd4a3BzcmtzdG5haHRtZ2hyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTU1ODEwNDQsImV4cCI6MjExMTE1NzA0NH0.kK4K_H3rSgZOEs_ErRKFcKGTqc0Td2zvdeVCgjTjT6U"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ตั้งค่า ONNX Runtime ให้ประหยัด RAM สูงสุด (ไม่จองหน่วยความจำล่วงหน้า)
sess_opts = ort.SessionOptions()
sess_opts.enable_mem_pattern = False
sess_opts.enable_cpu_mem_arena = False
sess_opts.intra_op_num_threads = 1

# ค้นหาไฟล์โมเดลจดจำใบหน้าของ buffalo_s ที่โหลดไว้ตอน Build
model_dir = "/root/.insightface/models/buffalo_s"
rec_files = glob.glob(os.path.join(model_dir, "*rec*.onnx")) or glob.glob(os.path.join(model_dir, "w600k*.onnx"))
rec_session = ort.InferenceSession(rec_files[0], sess_options=sess_opts, providers=["CPUExecutionProvider"])
rec_input_name = rec_session.get_inputs()[0].name

# ใช้ OpenCV Haar Cascade ตรวจจับกรอบใบหน้า (กิน RAM แค่ ~2 MB)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_face_embedding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # เลือกใบหน้าที่ใหญ่ที่สุดในภาพ
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    # ขยายขอบเล็กน้อยให้ครอบคลุมโครงหน้าเหมือน InsightFace
    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img.shape[1], x + w + pad_w), min(img.shape[0], y + h + pad_h)
    
    face_crop = cv2.resize(img[y1:y2, x1:x2], (112, 112))
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = (face_rgb - 127.5) / 127.5
    blob = np.transpose(face_norm, (2, 0, 1))[np.newaxis, ...]

    emb = rec_session.run(None, {rec_input_name: blob})[0][0]
    norm = np.linalg.norm(emb)
    return (emb / norm).tolist() if norm > 0 else emb.tolist()

@app.get("/")
def home():
    return {"status": "AVERB AI Face API is running!"}

@app.get("/events")
def get_events():
    res = supabase.rpc("get_events").execute()
    events = [row["event_name"] for row in res.data] if res.data else []
    return {"events": events}

@app.post("/search")
async def search_face(file: UploadFile = File(...), event_name: str = Form("ALL")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "ไม่สามารถอ่านไฟล์รูปภาพได้"}

    query_emb = get_face_embedding(img)
    if query_emb is None:
        return {"error": "ไม่พบใบหน้าในรูปที่คุณอัปโหลด กรุณาใช้รูปหน้าตรงที่ชัดเจน"}

    res = supabase.rpc("match_faces", {
        "query_embedding": query_emb,
        "match_threshold": 0.40,
        "match_count": 100,
        "filter_event": event_name
    }).execute()

    matches = []
    seen_files = set()
    for row in (res.data or []):
        fid = row["drive_file_id"]
        if fid in seen_files or "." in fid:
            continue
        seen_files.add(fid)
        matches.append({
            "file_id": fid,
            "event_name": row["event_name"],
            "similarity": round(row["similarity"] * 100, 1),
            "thumb_url": f"https://drive.google.com/thumbnail?id={fid}&sz=w600",
            "download_url": f"https://drive.google.com/uc?export=download&id={fid}"
        })

    return {"matches": matches}
