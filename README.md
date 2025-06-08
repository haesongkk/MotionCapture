# MotionCapture

## 📌 프로젝트 개요
영상 속 인물의 포즈를 불러와 실시간으로 3D 캐릭터의 포즈를 잡는다.


### 🛠 개발 환경
- Language: Python 3.9.13
- Libraries: mediapipe, numpy, opencv, PyOpenGL, ...
- Platform: Window 11

### 🔗 리소스 출처
- 3D어셋: [mixamo](https://www.mixamo.com/#/)
- 동영상: [youtube](https://www.youtube.com/shorts/yDmeZCPYNZ8)

---

## 🔍 결과물

[![시연 영상](https://img.youtube.com/vi/_EFuHFMcCv8/0.jpg)](https://www.youtube.com/watch?v=_EFuHFMcCv8)

---

## 🔧 개선사항
- 깊이 값을 정확하게 읽지 못함
- 옆모습과 뒷모습의 경우 정확도가 매우 낮아짐
- 모든 관절의 방향을 설정하지 않음 (손, 발, 목 등 안 움직임)
- mixamo 리깅만 호환됨 (본 이름 하드코딩)
- 의도치 않은 거울 모드
- 코드 난잡...
