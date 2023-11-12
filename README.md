# Emopic-AI
Emopic-AI 저장소에는 Emopic 프로젝트에 사용되는 Multi-Label Classification Model과 Captioning 모델을 serving 하는 코드가 작성되어 있습니다. 현재 소스코드는 redis를 queue로 사용하는 비동기 추론 패턴으로 구성되어 있습니다. 동기 추론 패턴을 확인하시려면 [링크](./ai-serving-sync/README.md)를 확인해 주세요


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
![Redis](https://img.shields.io/badge/redis-7.2.3-red)

### 배포
![docker](https://img.shields.io/badge/docker-blue)
![docker-compose](https://img.shields.io/badge/docker_compose-3.8-blue)

## 준비하기

- 서버를 시작하기전에 [훈련된 모델](#참고-사이트)이 필요합니다. 모델들을 다운받거나 참고사이트를 확인해 훈련시켜주세요.

- 로컬에서 서버를 실행시키기 보단 도커를 사용해 서버를 실행시키는 것을 추천합니다.

### Image docker
- Emopic AI는 미리 만들어진 docker-image를 제공합니다
    - [pre-build-ai-image](https://hub.docker.com/r/emopic/ai-serving-internal)
    - [pre-build-python-image](https://hub.docker.com/r/emopic/deploy-python)


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
- 총 9개의 컨테이너가 동작합니다. 내부 컨테이너간에 통신은 grpc 프로토콜을 사용합니다. 

    ```
    1. nginx                 (reverse proxy)
    2. redis                 (job queue)
    3. router                (fastapi server)
    4. class-client          (class 서버에 요청을 보내는 client)
    5. class-server          (classsification을 수행하는 ML server)
    6. caption-client        (caption 서버에 요청을 보내는 client)
    7. caption-server        (captioning을 수행하는 ML server)
    8. mysql-class-client    (classification 결과를 DB에 전송하는 client)
    9. mysql-caption-client  (captioning 결과를 DB에 전송하는 client)
    ```

#### 1. 레포지토리 클론 
```shell 
git clone https://github.com/Memento-Makers/Emopic-AI.git
```

#### 2. 학습된 모델 resource 폴더로 이동
- 다운로드 받거나 학습시킨 모델을 각 폴더(caption, classification)의 resources 폴더에 넣어 주세요.
- 밑의 .env파일에 사용하려는 모델 경로를 작성해주세요.


#### 3. 설정 파일 변경
- config 폴더 안의 .env 파일을 원하는 값으로 수정해주세요.
- database에 대한 정보는 사용하시려는 db 정보와 일치시켜 주세요 (mysql이 아닌 다른 db를 사용하신다면 관련된 라이브러리 설치와 src/database안의 소스코드를 db에 맞게 수정해주세요)

- env
    <details>

    ```
    REDIS_HOST=redis-queue
    REDIS_PORT=6379
    REDIS_DB=0
    REDIS_DECODE_RESPONSES=True

    API_TITLE=Emopic Inference Server
    API_DESCRIPTION=emopic ML serving 
    API_VERSION=0.1

    CAPTION_PROCESS_NUM=2
    CLASS_PROCESS_NUM=2

    MYSQL_HOST=host.docker.internal
    MYSQL_PORT=3306
    MYSQL_USER=root
    MYSQL_PASSWORD=
    MYSQL_DB=emopic

    CLASS_MODEL_HOST=class-server
    CLASS_MODEL_PORT=8600
    CAPTION_MODEL_HOST=caption-server
    CAPTION_MODEL_PORT=8500

    CLASS_MODEL_PATH=resources/coco_XL_model.pth
    CAPTION_MODEL_PATH=resources/rf_model.pth
    CAPTION_TOKEN_PATH=demo_coco_tokens.pickle

    # Log 
    CAPTION_SERVER_LOG_PATH=/app/log/caption_server.log
    CLASS_SERVER_LOG_PATH=/app/log/class_server.log
    CAPTION_CLIENT_LOG_PATH=/app/log/caption_client.log
    CLASS_CLIENT_LOG_PATH=/app/log/class_client.log
    MYSQL_CLASS_CLIENT_LOG_PATH=/app/log/mysql_class_client.log
    MYSQL_CAPTION_CLIENT_LOG_PATH=/app/log/mysql_caption_client.log

    DEEPL_AUTH_KEY=your_key
    ```
    </details>
    <div markdown="1">

#### 4. 로그 파일 경로 설정
- 기본적으로 docker-compose.yml 파일이 있는 폴더에 log 폴더를 생성해주시고 각 컨테이너 별로 로그가 저장될 수 있게 컨테이너 이름별로 폴더를 만들어 주세요.
(docker-compose.yml 파일을 참고해 주세요)

#### 5. 개발 환경 실행
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

## [기여하기](docs/contribute.md)
