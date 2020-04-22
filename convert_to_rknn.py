from rknn.api import RKNN

#create RKNN object
rknn = RKNN(verbose=True)

#Load model
print('--> Loading model') 
ret = rknn.load_tensorflow(tf_pb='./convert/freeze.pb',
                           inputs=['Placeholder'],
                           outputs=['Softmax'],
                           input_size_list=[[2304]])

#오류 발생시 오류 메시지 출력
if ret !=0:
  print('Load failed!')
  exit(ret)
#Load 완료시 'done'출력
print('done')

#Build model
print('--> Building model') #빌드하는 중임을 알려주는 출력
ret = rknn.build(do_quantization=False)
#오류 발생시 오류 메시지 출력
if ret !=0:
  print('Build failed!')
  exit(ret)
#Build 완료시 'done'출력
print('done')

#Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn('./emotion.rknn') #다음과 같은 이름으로 저장됩니다.
#오류 발생시 오류 메시지 출력
if ret != 0:
  print('Export failed!')
  exit(ret)
#export 완료시 'done'출력
print('done')

