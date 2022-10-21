import numpy as np
import tensorflow as tf
import cv2

 
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('C:\\Users\\T-GAMER\\Desktop\\workspace\\poc_posicionamento_real\\dataset\\videos\\teste_1.mp4')
# capture first frame
ret, frame = cap.read()

model = tf.keras.models.load_model("modelos/modelotf/")
print("model loaded") 
layer_name = "conv2d"
extractor = tf.keras.Model(inputs=model.inputs,outputs=model.get_layer(layer_name).output)
print("extractor")


# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# VideoWriter (const String &filename, int fourcc, double fps, Size frameSize, bool isColor=true)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('project.avi', fourcc, 30, (432//2, 768//2))
print(frame.shape)


img_arr = []

i = 0
# Loop until the end of the video
while (cap.isOpened() and ret):
    i += 1
    if i % 2 == 0 or i < 100:
      continue
                          
    model_input = cv2.resize(frame, (224, 224), fx = 0, fy = 0)
    model_input = cv2.rotate(model_input, cv2.ROTATE_180) 
    model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)           
        
    model_out = model(tf.expand_dims(model_input, 0))
    point = tf.argmax(model_out[0]).numpy()
    print(point)
    
    features = extractor(tf.expand_dims(model_input, 0))[0]
    
    
    # Display the resulting frame
    cv2.imshow('Frame', cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR))
    
    for chanel in range(16):    
      img = cv2.resize(features[..., chanel].numpy(), (224,224))
      cv2.imshow('Chanel{}{}'.format(chanel,features.shape), img)
      
    
    
    
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame = cv2.resize(frame, (432//2, 768//2), fx = 0, fy = 0)    
    frame = cv2.rotate(frame, cv2.ROTATE_180) 
    cv2.putText(img=frame, text='point: '+str(point), org=(5, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)    
    #img_arr.append(frame)
    out.write(frame)
                       
    # Capture frame-by-frame
    ret, frame = cap.read() 
    


for i in range(len(img_arr)):
    out.write(img_arr[i])


# Release everything if job is finished
out.release()    
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

