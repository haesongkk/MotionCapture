# MotionCapture

## 📌 프로젝트 개요
영상 속 인물의 포즈를 불러와 실시간으로 3D 캐릭터의 포즈를 잡는다.  


인물의 관절을 불러오기 위해 mediapipe를 이용했다.  
3D 어셋은 glb 파일로 변환하여 불러왔으며,  
openGL과 glsl 코드를 이용해 렌더링했다.  

### 🛠 개발 환경
- Language: Python 3.9.13
- Libraries: mediapipe, numpy, opencv, PyOpenGL, ... (requirements.txt)
- Platform: Window 11

python 3.9.13 환경에서  
`pip install -r requirements.txt` 라이브러리 설치 후  
`python main.py`를 통해 실행한다.

> 실행 중 강제 종료 (ctrl + c) 로만 종료가 가능하며
> 영상이 끝나면 자동으로 종료된다.

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
- 튀는 프레임을 잡기위해 보정을 했는데, 동작이 느릿해짐
