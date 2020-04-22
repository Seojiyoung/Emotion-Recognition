import cv2
import numpy as np
import sys
import tensorflow as tf

from model import predict, image_to_tensor, deepnn
from rknn.api import RKNN

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def format_image(image):
  #이미지를 Gray로 바꿈
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #input이미지에서 크기가 다른 object를 검출하는 함수
    """
    image는 검출하고자 하는 이미지
    scaleFactor는 이미지 피라미드에서 사용되는 scale factor
    minNeighbors는 여러 스케일의 크기에서 minHeighbors횟수 이상 검출된 object는 
    valid하게 검출할 때 사용
    """
    faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3,minNeighbors = 5)

  # 이미지에 얼굴이 없으면 none 리턴
  if not len(faces) > 0:
    return None, None

  # faces리스트의 첫번째를 max_are_face로 설정
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face

  # face를 이미지로 바꿈
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]

  # 이미지 사이즈를 48,48로 줄임
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)

  #오류 메시지 출력
  except Exception:
    print("[+} Problem during resize")
    return None, None

  #이미지, 가장 큰 facd object 리턴
  return  image, face_coor

def face_dect(image):
  """
  Detecting faces in image
  :param image:
  :return:  the coordinate of max face
  """

  # 이미지를 Gray로 변환
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # 다양한 face object 찾는 함수
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )

  # face 검출이 안 되면 None 리턴
  if not len(faces) > 0:
    return None

  # 가장 큰 face object 찾기
  max_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_face[2] * max_face[3]:
      max_face = face
  face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]

  # 이미지 resize (48,48) 사이즈로 resize
  try:
    image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+} Problem during resize")
    return None
  return face_image

# 이미지 resize
def resize_image(image, size):
  try:
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
  except Exception:
    print("+} Problem during resize")
    return None
  return image


def main():
  
  #create RKNN object
  rknn = RKNN(verbose=True)

  #Direct Load RKNN Model
  rknn.load_rknn('./emotion.rknn') #만들어진 rknn을 로드
  print('--> load success') #성공 메세지 출력

  result = None

  #이미지 읽기
  input_image = cv2.imread('./data/image/happy.jpg', cv2.IMREAD_COLOR)

  #esize한 이미지, 가장 큰 얼굴 object
  detected_face, face_coor = format_image(input_image)

  #탐지된 이미지가 있다면,
  if detected_face is not None:
    #image를 tenxor로 변환 & float32로 변환 (rknn이 float64는 지원하지 않음)
    "tensor 사이즈는 (1,2304), detected_face는 48X48"
    tensor = image_to_tensor(detected_face).astype(np.float32)
    
    #init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    #오류 메세지 출력
    if ret != 0:
      print('Init runtime environment failed')

    #rknn 모델 실행
    result = rknn.inference(inputs=[tensor])
    print('run success')
    
    #list를 array로 변환
    #result는 감정 예측 배열
    result = np.array(result)
   
    #result가 존재하면   
    if result is not None:
      #감정 배열이 7개의 값을 가지므로 range(7)의 범위를 가짐
      for i in range(7):
        #감정 배열 중 1인 값이 있다면,
        if result[0][0][i] == 1:
          #감정 예측 메세지 출력
          print('당신의 감정은 '+EMOTIONS[i]+'입니다.')

if __name__ == '__main__':
  main()

