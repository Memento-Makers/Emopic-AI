# Emopic-AI
Emopic-AI 저장소에는 Emopic 프로젝트에 사용되는 Multi-Label Classification Model과 Captioning 모델을 serving 하는 코드가 작성되어 있습니다. 
(ai-serving-sync 폴더에 있는 코드는 API서버와 동기적으로 작동합니다.)

## Table Of content

- [기술 스택](#기술-스택)
- [준비 하기](#준비하기)
- [실행 하기](#실행하기)
- [참고사이트](#참고-사이트)
- [기여 하기](#기여하기)


## 기술 스택

### 언어 

![python](https://img.shields.io/badge/python-3.8-blue)

### 라이브러리 
![python](https://img.shields.io/badge/pillow-10.1.0-green)

### 프레임워크
![PyTorch](https://img.shields.io/badge/pytorch-2.0.0+cu118-EE4C2C)
![FastAPI](https://img.shields.io/badge/fastapi-0.104.1-005571)

### 배포
![docker](https://img.shields.io/badge/docker-blue)
![docker-compose](https://img.shields.io/badge/docker_compose-3.8-blue)

## 준비하기

- 서버를 시작하기전에 [훈련된 모델](#참고-사이트)이 필요합니다. 모델들을 다운받거나 참고사이트를 확인해 훈련시켜주세요.

- 로컬에서 서버를 실행시키기 보단 도커를 사용해 서버를 실행시키는 것을 추천합니다.

### Image docker
- Emopic AI는 미리 만들어진 docker-image를 제공합니다
    - [pre-build-image](https://hub.docker.com/r/emopic/ai-serving)

## 실행하기

### Build docker

```
$ docker build -t {image-name} \
    --build-args CUDA_VER=11.8.0 \
    --build-args CUDA_PATH_VER=11.8 \
    --build-args CUDNN_VER=8 \
    --build-args UBUNTU_VER=20.04 \
    --build-args PYTHON_VER=38
```

### Docker Compose

- 필요한 소스코드는 docker-compose의 volume을 이용해 컨테이너 안으로 넣습니다. 그래서 컨테이너를 띄운 뒤에도 소스코드를 수정할 수 있습니다.


#### 1. 레포지토리 클론 
```shell 
git clone https://github.com/Memento-Makers/Emopic-AI.git
cd ai-serving-sync
```

#### 2. 학습된 모델 resource 폴더로 이동
- 다운로드 받거나 학습시킨 모델을 각 폴더(caption, classification)의 resources 폴더에 넣어 주세요.


#### 3. 설정 파일 변경
- caption, classification 폴더 안의 .env 파일을 원하는 값으로 수정해주세요.
- 특히 모델 경로에 모델 이름을 변경해주셔야 합니다.
- env
    - caption
        ```
        port=8000
        host=0.0.0.0
        model_path= resources/rf_model.pth
        token_path= config/demo_coco_tokens.pickle
        ```
    - classification
        ```
        port=8001
        host=0.0.0.0
        model_path= resources/coco_XL_model.pth
        ```

#### 4. 개발 환경 실행
```shell
docker compose up -d
``` 

## 참고 사이트
- Mutli-Label Classification
    - [pretrained model](https://github.com/Alibaba-MIIL/ML_Decoder/blob/main/MODEL_ZOO.md)
    - https://github.com/Alibaba-MIIL/ML_Decoder
- Caption
    - [pretrained model](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3)
    - https://github.com/jchenghu/ExpansionNet_v2

## [기여하기](../docs/contribute.md)
