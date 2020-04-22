# 표정 인식
### ICT인턴을 진행하며 만든 REX보드 예제입니다. (RKNN 모델로 변환하여 이용)
angry, disgusted, fearful, happy, sad, suprised, neutral 총 7개의 표정을 인식<br>
Facial Expression Recognition(FER) 2013에서 우수한 정확도를 기록한 모델<br>
Tensorflow 기반으로 짜여진 CNN네트워크를 사용
<br>

필요한 라이브러리 버전
-
- Python (>= 3.3)
- Tensorflow (>= 1.1.0)
- OpenCV (python3-version)

사용한 라이브러리 버전
-
- Python 3.5.2
- rknn에서 요구하는 라이브러리 그대로 사용

파일 설명
-
- convert : rknn으로 바꾸기 위해 필요한 ckpt, pbtxt, frozen pb 파일
- data : haarcascade_files는 얼굴 인식, image는 예제 이미지

RKNN모델 변환
-
	python convert_to_rknn.py

코드 실행
-
    python demo.py

<img src="/data/image/happy.jpg" width="30%" height="30%">
<br>
<img src="/data/image/output_happy.jpg">

인용
-
https://github.com/xionghc/Facial-Expression-Recognition
