import os
import gc
import glob
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from insightface.model_zoo import SCRFD
from insightface.utils import face_align
from supabase import create_client

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://xxngxkpsrkstnahtmghr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh4bmd4a3BzcmtzdG5haHRtZ2hyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTU1ODEwNDQsImV4cCI6MjExMTE1NzA0NH0.kK4K_H3rSgZOEs_ErRKFcKGTqc0Td2zvdeVCgjTjT6U"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

sess_opts = ort.SessionOptions()
sess_opts.enable_mem_pattern = False
sess_opts.enable_cpu_mem_arena = False
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1

model_dir = "/root/.insightface/models/buffalo_l"
rec_files = glob.glob(os.path.join(model_dir, "w600k*.onnx"))
rec_session = ort.InferenceSession(rec_files[0], sess_options=sess_opts, providers=["CPUExecutionProvider"])
rec_input_name = rec_session.get_inputs()[0].name

det_files = glob.glob(os.path.join(model_dir, "det_*.onnx"))
det_session = ort.InferenceSession(det_files[0], sess_options=sess_opts, providers=["CPUExecutionProvider"])
detector = SCRFD(model_file=det_files[0], session=det_session)
detector.prepare(ctx_id=-1, input_size=(320, 320))

def get_face_embedding(img):
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    bboxes, kpss = detector.detect(img, max_num=1, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return None

    if kpss is not None and len(kpss) > 0:
        face_crop = face_align.norm_crop(img, landmark=kpss[0], image_size=112)
    else:
        x1, y1, x2, y2 = bboxes[0][:4].astype(int)
        face_crop = cv2.resize(img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)], (112, 112))

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = (face_rgb - 127.5) / 127.5
    blob = np.transpose(face_norm, (2, 0, 1))[np.newaxis, ...]

    emb = rec_session.run(None, {rec_input_name: blob})[0][0]
    norm = np.linalg.norm(emb)
    result = (emb / norm).tolist() if norm > 0 else emb.tolist()
    
    del img, face_crop, face_rgb, face_norm, blob
    gc.collect()
    return result

@app.get("/")
def home():
    return {"status": "AVERB AI Face API (buffalo_l) is running!"}

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
    del contents, nparr
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
        if fid in seen_files:
            continue
        seen_files.add(fid)
        matches.append({
            "file_id": fid,
            "event_name": row["event_name"],
            "similarity": round(row["similarity"] * 100, 1),
            "thumb_url": f"https://lh3.googleusercontent.com/d/{fid}=w600",
            "download_url": f"https://drive.google.com/uc?export=download&id={fid}"
        })

    gc.collect()
    return {"matches": matches}
