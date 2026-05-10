# MotionCapture

영상 속 사람의 포즈를 추정하고, 추정된 관절 방향을 3D 캐릭터 모델에 적용하여 렌더링하는 실험 프로젝트입니다.

MediaPipe로 영상 프레임의 인체 관절을 추정하고, GLB 형식의 스킨드 캐릭터 모델을 직접 로드한 뒤, OpenGL/GLSL 기반 렌더링 파이프라인에서 캐릭터 포즈를 갱신합니다.

## 실행 화면

[![시연 영상](https://img.youtube.com/vi/_EFuHFMcCv8/maxresdefault.jpg)](https://www.youtube.com/watch?v=_EFuHFMcCv8)

> 위 이미지를 클릭하면 시연 영상을 확인할 수 있습니다.

## 파이프라인

```text
Input Video
    ↓
Frame Capture
    ↓
MediaPipe Pose Estimation
    ↓
Landmark Smoothing
    ↓
Joint Direction Mapping
    ↓
Mixamo Bone Mapping
    ↓
GLB Mesh / Skinning Data Loading
    ↓
OpenGL / GLSL Rendering
    ↓
Animated 3D Character
````

### 1. 영상 프레임 입력

OpenCV를 사용하여 입력 영상을 프레임 단위로 읽습니다.
각 프레임은 MediaPipe Pose 모델에 전달되어 인체 관절 추정에 사용됩니다.

### 2. 인체 관절 추정

MediaPipe Pose를 이용해 영상 속 인물의 주요 관절 위치를 추정합니다.
추정된 landmark는 `(x, y, z)` 좌표 형태로 저장되며, 이후 3D 캐릭터의 본 방향을 계산하는 데 사용됩니다.

### 3. 관절 데이터 보정

프레임마다 추정되는 관절 좌표는 흔들림이 발생할 수 있으므로, 이전 프레임과 현재 프레임을 보간하고 일정 프레임 구간의 평균을 사용해 움직임을 완화했습니다.

이를 통해 갑작스럽게 튀는 프레임을 줄이고, 캐릭터 포즈가 조금 더 안정적으로 갱신되도록 구성했습니다.

### 4. Mixamo 본 구조 매핑

MediaPipe에서 얻은 관절 위치를 그대로 사용할 수는 없기 때문에, 관절 간 시작점과 끝점을 이용해 방향 벡터를 계산합니다.

이후 해당 방향 벡터를 Mixamo 리깅의 본 이름과 매핑합니다.

예를 들어 다음과 같은 방식으로 관절 방향을 본에 연결합니다.

```text
MediaPipe Shoulder → Elbow  → mixamorig:LeftArm / mixamorig:RightArm
MediaPipe Elbow → Wrist     → mixamorig:LeftForeArm / mixamorig:RightForeArm
MediaPipe Hip → Knee        → mixamorig:LeftUpLeg / mixamorig:RightUpLeg
MediaPipe Knee → Ankle      → mixamorig:LeftLeg / mixamorig:RightLeg
```

### 5. GLB 모델 로드

`pygltflib`를 사용해 GLB 파일을 읽고, 렌더링과 스키닝에 필요한 데이터를 직접 추출합니다.

로드하는 주요 데이터는 다음과 같습니다.

* Vertex Position
* Normal
* Texture Coordinate
* Index
* Joint Index
* Skin Weight
* Texture
* Node Hierarchy
* Inverse Bind Matrix

### 6. 스키닝 행렬 계산

GLB 모델의 노드 계층 구조와 inverse bind matrix를 이용해 각 본의 변환 행렬을 계산합니다.
프레임마다 갱신된 관절 방향을 기준으로 본의 local transform을 수정하고, 최종 joint matrix를 셰이더로 전달합니다.

### 7. OpenGL / GLSL 렌더링

OpenGL을 사용해 VAO, VBO, EBO를 구성하고, GLSL 셰이더에서 joint index와 weight를 이용해 스키닝된 정점 위치를 계산합니다.

이를 통해 영상 프레임이 진행될 때마다 3D 캐릭터가 추정된 포즈를 따라 움직이도록 렌더링합니다.

## 기술 스택

### Language

* Python 3.9.13
* GLSL

### Computer Vision

* OpenCV
* MediaPipe
* NumPy

### 3D / Graphics

* PyOpenGL
* GLSL Vertex / Fragment Shader
* pygltflib
* pyrr
* pyglet
* PIL

### 3D Asset

* GLB
* Mixamo Rigging

## 프로젝트 구조

```text
MotionCapture/
│
├── main.py              # 프로그램 진입점
├── PoseEstimator.py     # 영상 입력 및 MediaPipe 기반 포즈 추정
├── GLBLoader.py         # GLB 모델, 메시, 텍스처, 스키닝 데이터 로드
├── Model.py             # 모델 데이터, 본 계층 구조, 스키닝 행렬 계산 및 렌더링
├── Shader.py            # GLSL 셰이더 로드 및 컴파일
├── Process.py           # 실행 흐름 보조 처리
│
├── shader.vert          # Vertex Shader
├── shader.frag          # Fragment Shader
│
├── requirements.txt     # Python 의존성 목록
└── README.md
```

### 주요 파일 설명

#### `main.py`

프로그램의 실행 진입점입니다.
포즈 추정기, GLB 로더, 모델, 셰이더를 초기화하고, 매 프레임 포즈 추정 결과를 캐릭터 모델에 전달해 렌더링합니다.

#### `PoseEstimator.py`

OpenCV로 영상을 읽고, MediaPipe Pose를 사용해 인체 관절을 추정합니다.
추정된 landmark를 Mixamo 본 구조에 맞는 관절 방향 데이터로 변환합니다.

또한 프레임 간 튀는 값을 줄이기 위해 이전 프레임과 현재 프레임을 보간하고, 일정 프레임의 평균을 사용하는 방식으로 관절 데이터를 보정합니다.

#### `GLBLoader.py`

GLB 파일에서 렌더링에 필요한 메시 데이터를 직접 추출합니다.

* position
* normal
* texcoord
* index
* joint
* weight
* texture
* node hierarchy
* inverse bind matrix

스킨드 메시 렌더링에 필요한 기초 데이터를 구성하는 역할을 합니다.

#### `Model.py`

로드된 메시와 본 계층 구조를 바탕으로 실제 렌더링과 스키닝 행렬 계산을 담당합니다.

프레임마다 전달된 관절 방향 데이터를 이용해 각 본의 transform을 갱신하고, 계산된 joint matrix를 셰이더에 전달하여 캐릭터 포즈를 변경합니다.

#### `Shader.py`

GLSL 셰이더 파일을 읽고, OpenGL에서 사용할 수 있도록 컴파일 및 링크합니다.

#### `shader.vert`

스키닝 연산을 수행하는 Vertex Shader입니다.
각 정점의 joint index와 weight를 사용해 본 변환을 적용합니다.

#### `shader.frag`

텍스처 색상을 출력하는 Fragment Shader입니다.

## 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/haesongkk/MotionCapture.git
cd MotionCapture
```

### 2. Python 가상환경 생성

```bash
python -m venv .venv
```

Windows PowerShell 기준:

```bash
.venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 리소스 준비

실행을 위해 다음 리소스가 필요합니다.

* 입력 영상 파일
* Mixamo 리깅 구조를 가진 GLB 캐릭터 모델

현재 코드는 특정 파일 경로를 기준으로 작성되어 있을 수 있으므로, 실행 전 `main.py` 내부의 영상 경로와 GLB 모델 경로를 자신의 환경에 맞게 수정해야 합니다.

예시:

```python
video_path = "your_video.mp4"
glb_path = "your_character.glb"
```

### 5. 실행

```bash
python main.py
```

실행 중에는 OpenCV 창에 입력 영상의 포즈 추정 결과가 표시되고, 별도의 OpenGL 창에서 3D 캐릭터가 렌더링됩니다.

> 현재 버전에서는 실행 중 강제 종료가 필요한 경우 `Ctrl + C`를 사용할 수 있습니다.
> 입력 영상이 끝나면 프로그램이 종료됩니다.

## 현재 한계 및 개선 방향

이 프로젝트는 영상 기반 포즈 추정 결과를 3D 캐릭터에 연결하는 실험 프로젝트이며, 다음과 같은 한계가 있습니다.

* MediaPipe의 깊이 값이 실제 3D 공간의 깊이와 정확히 일치하지 않음
* 옆모습이나 뒷모습 영상에서는 포즈 추정 정확도가 낮아질 수 있음
* 손, 발, 목 등 일부 관절 방향은 아직 완전하게 적용하지 않음
* 현재는 Mixamo 리깅 이름을 기준으로 본 매핑이 구성되어 있어 다른 리깅 구조와의 호환성이 낮음
* 관절 보정을 적용하면서 움직임이 다소 느리게 반영될 수 있음

향후에는 다음 방향으로 개선할 수 있습니다.

* 본 이름 하드코딩 제거 및 리깅 매핑 테이블 분리
* 손, 발, 목 등 세부 관절 적용
* 포즈 보정 필터 개선
* 카메라 방향과 좌표계 보정
* 다양한 GLB 모델에 대응할 수 있는 범용 구조로 확장

## 리소스 출처

* 3D Character Asset: [Mixamo](https://www.mixamo.com/)
* Input Video: [YouTube](https://www.youtube.com/)
