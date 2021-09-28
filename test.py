import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2


np.set_printoptions(suppress=True)
guven = 0.7
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap  = cv2.VideoCapture(0)
while(cap.isOpened()):

    ret, frame = cap.read()

    if ret==True:
        size = (224, 224)
        img = Image.fromarray(frame)
        image = ImageOps.fit(img, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        for numara in prediction:
            maskesiz = numara[0]
            maskeli = numara[1]

            a = round(float(maskeli),3)
            b = round(float(maskesiz),3)

            if maskeli > guven:
                cv2.putText(frame,f"maskeli : {a}",(10,25),cv2.FONT_ITALIC,1,(255,0,0),0)
            else:

                cv2.putText(frame, f"maskesiz: {b}", (10, 25), cv2.FONT_ITALIC, 1, (0, 0, 255), 0)

        cv2.resize(frame,(480,480))

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


