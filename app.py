import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
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

# ใช้รุ่น buffalo_s เพื่อประหยัด RAM บน Render Free (512MB) และโหลดเฉพาะโมเดลตรวจจับ+จดจำใบหน้า
face_app = FaceAnalysis(
    name="buffalo_s",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

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

    faces = face_app.get(img)
    if not faces:
        return {"error": "ไม่พบใบหน้าในรูปที่คุณอัปโหลด กรุณาใช้รูปหน้าตรงที่ชัดเจน"}

    query_emb = faces[0].embedding.tolist()
    res = supabase.rpc("match_faces", {
        "query_embedding": query_emb,
        "match_threshold": 0.45,
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
