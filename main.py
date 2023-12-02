from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import asyncio
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from insightface.app import FaceAnalysis
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
from pathlib import Path

# InsightFace 초기화
model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()

# 정적 파일(HTML)을 제공
app.mount("/", StaticFiles(directory="templates", html=True), name="templates")

# Templates 초기화
templates = Jinja2Templates(directory="templates")

#웹캠에서 프레임을 읽어오고, InsightFace를 사용하여 얼굴 감지
async def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # InsightFace를 사용하여 얼굴 감지
        faces = model.get(frame)

        # 초록색 프레임 표시
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

        yield cv2.imencode('.jpg', frame)[1].tobytes()
        await asyncio.sleep(0.05)
    cap.release()


# index.html을 렌더링하는 엔드 포인트
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#비디오 프레임 스트리밍
@app.get("/video_feed")
async def video_feed():
    async def video_stream():
        async for frame in get_frame():
            yield b"data: " + frame + b"\n\n"
    return StreamingResponse(video_stream(), media_type="text/event-stream")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://google.com/"],  # 실제 서비스 시에는 허용하려는 도메인을 명시합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/save_face")
async def save_face(file: UploadFile = File(...)):
    contents = await file.read()
    # Save the image to the specified path
    with open("captured_face.jpg", "wb") as f:
        f.write(contents)
    return {"message": "Face saved successfully"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
