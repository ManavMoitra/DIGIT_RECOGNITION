import cv2
import numpy as np
import os
import tensorflow as tf


model = tf.keras.models.load_model('C:/Users/KIIT/Desktop/digitrecogniser/mnist.h5')
print(model.summary())
a=np.ones((300,300,3),dtype=np.uint8)*255
wname='Shapes'
cv2.namedWindow(wname)



def predict(image):
    
    image=image[:,:,0]
    input=cv2.resize(image,(28,28)).reshape((28,28,1)).astype('float32')/255
    return np.argmax(model.predict(np.array([input])))




def shape(event,x,y,flags,params):
    global ix, iy,drawing
      
    if event == cv2.EVENT_LBUTTONDOWN: 
        drawing = True
        ix = x+10
        iy = y+10             
              
    elif event == cv2.EVENT_MOUSEMOVE: 
        if drawing == True: 
            cv2.rectangle(a, pt1 =(x, y), 
                          pt2 =(x+10, y+10), 
                          color =(0, 0, 0), 
                          thickness =5) 
      
    elif event == cv2.EVENT_LBUTTONUP: 
        drawing = False


cv2.setMouseCallback(wname,shape)    

while True:    
    cv2.imshow(wname,a)
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break
    elif(key==ord('c')):
        a[:,:]=255
    elif(key==ord('p')):
         image=a[100:300,100:300]
         result=predict(image)
         print("Predicted digit is: ",result)
       
cv2.destroyAllWindows()
