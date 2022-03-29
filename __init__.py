from tensorflow import keras
import cv2
import numpy as np
import os, sys
from gtts import gTTS
from flask import Flask,render_template
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
# def home():
#     return '''
#     <h1>안봐도 보이조 서비스 구현 웹 사이트입니다.</h1>
#     <p>테스트를 시작하시면 카메라 앱이 실행됩니다. 아무키나 누르시면 촬영이 됩니다. </p>
#     <a href="http://127.0.0.1:5000/test">테스트 시작</a>
#     '''
@app.route("/test")
def predicts():
    cap = cv2.VideoCapture(0)  # 카메라 지정

    if not cap.isOpened():  # 카메라가 제대로 불러지지 않으면 오류메세지 출력
        print('video capture failed')
        sys.exit()

    while True:
        ret, frame = cap.read()  # 카메라 읽기

        if not ret:  # 읽기 실패 시 오류 메세지 출력
            print('videos read failed')
            break

        window_name = "camera"
        cv2.imshow("camera", frame)  # 읽어온 카메라 영상을 출력
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(20) >= 0:  # s를 누르면 해당 순간의 이미지를 저장하고 종료
            cv2.imwrite("/workspace/can_flask/static/images/photo.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()  # 카메라 및 창 닫기

    img = cv2.imread('/workspace/can_flask/static/images/photo.jpg')  # 저장한 이미지 불러오기
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 타입변경
    img = cv2.resize(img, (224, 224)) / 255.0  # 모델에 맞는 input_shape로 리사이즈
    img = img.reshape((1,) + img.shape)  # 입력 데이터로 사용하기 위해 데이터 reshape
    # 모델 파일 불러와서 예측 수행
    model = keras.models.load_model('/workspace/can_flask/model/MobileNetV2_04.h5')
    pred = model.predict(img)

    # 인덱스로 상품명 추출
    class_dict = {0:'갈아만든배',
                 1:'레쓰비',
                 2:'마운틴듀',
                 3:'밀키스',
                 4:'스프라이트',
                 5:'칠성사이다',
                 6:'코카콜라',
                 7:'트로피카나망고',
                 8:'펩시콜라',
                 9:'환타오렌지'}

    pred_class = class_dict[np.argmax(pred, axis=1)[0]]

    text ='예측해보니 가장 가능성이 높은 항목은 '+pred_class+ ' 입니다'
    language = 'ko'
    speech = gTTS(text=text, lang=language, slow=False)
    speech.save('/workspace/can_flask/static/sound/text.mp3')
    # os.system('start text.mp3')

    return render_template('results.html', audiofile = 'sound/text.mp3',
                           value = pred_class,
                           image_file = 'images/photo.jpg' )



# if __name__ == '__main__':
#     app.run(debug=True)
    
if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port=80)