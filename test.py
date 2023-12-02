import cv2
from matplotlib import pyplot as plt
from insightface.app import FaceAnalysis

# InsightFace 초기화
model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# 웹캠에서 프레임 읽기
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 얼굴 감지
faces = model.get(frame)

# 감지된 얼굴 주위에 프레임 표시
for face in faces:
    # 적절한 키를 사용하여 얼굴 좌표 가져오기
    x, y, w, h = face.get('bbox', (0, 0, 0, 0))
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

# OpenCV BGR 이미지를 RGB로 변환하여 표시
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Face Detection')
plt.show()

# 사용이 끝난 후 VideoCapture 객체 해제
cap.release()
