# import libraries
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import platform


#----- Data and initializations
model_path = "./bin/rps_edgetpu.tflite"

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

classes = ['Paper', 'Rock', 'Scissor']
classes_dic ={'Papaer': 0, 'Rock': 0, 'Scissor': 0}

# input size model
input_size = 150

# initialize interpreter
interpreter = tflite.Interpreter(model_path,
    experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
interpreter.allocate_tensors()

# initialize camera and opencv variables
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX


#----- Some useful functions
def processData(frame, sz):
    """Convert BGR2RGB | resize to model dim | normalize"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (sz,sz))
    return frame 

def setInput(interpreter, data):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:, :] = data

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def output_tensor(interpreter):
    """Returns dequantized output tensor."""
    output_details = interpreter.get_output_details()[0]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)

def drawInfo(frame, dictionary):
    """Draw text and dynamic rectangles"""
    cv2.putText(masked, 'Paper', (10, 25), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(masked, 'Rock', (10, 55), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(masked, 'Scissor', (10, 85), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.rectangle(masked, (150,5), (150 + dictionary['Paper'], 25), (255,0,0), -1)
    cv2.rectangle(masked, (150,30), (150 + dictionary['Rock'], 50), (255,0,0), -1)
    cv2.rectangle(masked, (150,55), (150 + dictionary['Scissor'], 80), (255,0,0), -1)


#----- Main loop

while True:
    # capture frame
    _, frame = camera.read()

    #pre-process data

    img = processData(frame.copy(), input_size)
    # initialize interpreter
    setInput(interpreter, img)
    # invoke interpreter
    interpreter.invoke()
    # get output
    tflite_results = output_tensor(interpreter)

    # print info
    print("Prediction: {} Score: {}".format(classes[np.argmax(tflite_results)], np.amax(tflite_results)))

    # update dictionary
    classes_dic[classes[0]] = int(tflite_results[0] * 80)
    classes_dic[classes[1]] = int(tflite_results[1] * 50)
    classes_dic[classes[2]] = int(tflite_results[2] * 50)
    
    
    # prepare output frame
    # crop cricular
    mask = np.zeros(frame.shape[:2], dtype = 'uint8')
    cv2.circle(mask, (frame.shape[1] // 2, frame.shape[0] // 2), 180, 255, -1)
    masked = cv2.bitwise_and(frame, frame, mask = mask)
    
    # draw some stuf
    drawInfo(masked, classes_dic)

    # show frame
    cv2.imshow('Camera', masked)

    # check input. If 'esc' close window
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
