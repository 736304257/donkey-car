import os
import time
import numpy as np
from PIL import Image
import glob
from donkeycar.utils import rgb2gray
import cv2
import importlib.util
from threading import Thread
import argparse
import pickle

MODEL_NAME = '/home/pi/mycar/model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = float(0.5)
fenbianlv='160x120'
resW, resH = fenbianlv.split('x')
imW, imH = int(resW), int(resH)
# use_TPU = 'False'

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    # if use_TPU:
        # from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
# if use_TPU:
    # interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              # experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    # print(PATH_TO_CKPT)
# else:
    # interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
global frame_rate_calc
frame_rate_calc = 1
freq = cv2.getTickFrequency()

class BaseCamera:

    def run_threaded(self):
        return self.frame

class PiCamera(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20, vflip=False, hflip=False):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        
        resolution = (image_w, image_h)
        # initialize the camera and stream
        self.camera = PiCamera() #PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="rgb", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True
        self.image_d = image_d

        print('PiCamera loaded.. .warming camera')
        time.sleep(2)


    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)
        if self.image_d == 1:
            frame = rgb2gray(frame)
        return frame

    def update(self):
        global frame_rate_calc
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            t1 = cv2.getTickCount()
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            

            if self.image_d == 1:
                self.frame = rgb2gray(self.frame)
            
            # cv2.imshow('Object detector', self.frame)
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            frame2 = self.frame.copy()
            frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))#输入的图像
            input_data = np.expand_dims(frame_resized, axis=0)
            
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
                
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            # print("??????????????????????????????????????????????????",boxes)
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            counts=0
            no_danger = 0
            all_item = 0
            no_person = 0
            
            
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)and(classes[i]==0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    counts=counts+1
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    
                    #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    #object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    #label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    #label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    #cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                    with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "rb") as f:
                        try:
                            outputxmin = pickle.load(f)
                            outputymin = pickle.load(f)
                            outputxmax = pickle.load(f)
                            outputymax = pickle.load(f)
                            music_on = pickle.load(f)
                            f.close()
                        except EOFError:
                            return None
                    if(counts==1):
                        outputymin=ymin
                        outputxmin=xmin
                        outputxmax=xmax
                        outputymax=ymax
                        outputi=i
                    print("akjlhlsdhfaoergasgvjks",(outputxmin,outputymin),(outputxmax,outputymax))

                    with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "wb") as f:
                        pickle.dump(outputxmin, f)
                        pickle.dump(outputymin, f)
                        pickle.dump(outputxmax, f)
                        pickle.dump(outputymax, f)
                        pickle.dump(music_on, f)
                        f.close()
                        # pickle.dump(money_s, f)
                        # pickle.dump(money, f)
                        # pickle.dump(username, f)
                    # a = ((outputxmin,outputymin),(outputxmax,outputymax))
                    # return a
                    cv2.rectangle(frame2, (outputxmin,outputymin), (outputxmax,outputymax), (10, 255, 0), 2)
                     # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame2, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw
                            
                    # Draw framerate in corner of frame
                    cv2.putText(frame2,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                    # All the results have been drawn on the frame, so it's time to display it.
                    # cv2.imshow('Object detector', frame)

                    # Calculate framerate
                    t2 = cv2.getTickCount()
                    time1 = (t2-t1)/freq
                    frame_rate_calc= 1/time1
                    print("帧率",frame_rate_calc)
                
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)and(classes[i]==61)):
                    
                    with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "rb") as f:
                        try:
                            outputxmin = pickle.load(f)
                            outputymin = pickle.load(f)
                            outputxmax = pickle.load(f)
                            outputymax = pickle.load(f)
                            music_on = pickle.load(f)
                            f.close()
                        except EOFError:
                            return None
                    music_on = 1
                    with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "wb") as f:
                        pickle.dump(outputxmin, f)
                        pickle.dump(outputymin, f)
                        pickle.dump(outputxmax, f)
                        pickle.dump(outputymax, f)
                        pickle.dump(music_on, f)
                        f.close()
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)and(classes[i]!=61)):
                    no_danger += 1
                
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)and(classes[i]!=0)):
                    no_person += 1
                
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    all_item += 1
                
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",no_danger)
            if no_danger == all_item:
                with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "rb") as f:
                    try:
                        outputxmin = pickle.load(f)
                        outputymin = pickle.load(f)
                        outputxmax = pickle.load(f)
                        outputymax = pickle.load(f)
                        music_on = pickle.load(f)
                        f.close()
                    except EOFError:
                        return None
                music_on = 0
                with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "wb") as f:
                    pickle.dump(outputxmin, f)
                    pickle.dump(outputymin, f)
                    pickle.dump(outputxmax, f)
                    pickle.dump(outputymax, f)
                    pickle.dump(music_on, f)
                    f.close()
            if no_person == all_item:
                with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "rb") as f:
                    try:
                        outputxmin = pickle.load(f)
                        outputymin = pickle.load(f)
                        outputxmax = pickle.load(f)
                        outputymax = pickle.load(f)
                        music_on = pickle.load(f)
                        f.close()
                    except EOFError:
                        return None
                outputymin=-1
                outputxmin=-1
                outputxmax=imW
                outputymax=imH
                # music_on = 0
                
                with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "wb") as f:
                    pickle.dump(outputxmin, f)
                    pickle.dump(outputymin, f)
                    pickle.dump(outputxmax, f)
                    pickle.dump(outputymax, f)
                    pickle.dump(music_on, f)
                    f.close()

            # if the thread indicator variable is set, stop the thread
            if not self.on:
                break

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('Stopping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()

class Webcam(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate = 20, iCam = 0):
        import pygame
        import pygame.camera

        super().__init__()
        resolution = (image_w, image_h)
        pygame.init()
        pygame.camera.init()
        l = pygame.camera.list_cameras()
        print('cameras', l)
        self.cam = pygame.camera.Camera(l[iCam], resolution, "RGB")
        self.resolution = resolution
        self.cam.start()
        self.framerate = framerate

        # initialize variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True
        self.image_d = image_d

        print('WebcamVideoStream loaded.. .warming camera')

        time.sleep(2)

    def update(self):
        from datetime import datetime, timedelta
        import pygame.image
        while self.on:
            start = datetime.now()

            if self.cam.query_image():
                # snapshot = self.cam.get_image()
                # self.frame = list(pygame.image.tostring(snapshot, "RGB", False))
                snapshot = self.cam.get_image()
                snapshot1 = pygame.transform.scale(snapshot, self.resolution)
                self.frame = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), 90))
                if self.image_d == 1:
                    self.frame = rgb2gray(self.frame)

            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

        self.cam.stop()

    def run_threaded(self):
        return self.frame

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping Webcam')
        time.sleep(.5)


class CSICamera(BaseCamera):
    '''
    Camera for Jetson Nano IMX219 based camera
    Credit: https://github.com/feicccccccc/donkeycar/blob/dev/donkeycar/parts/camera.py
    gstreamer init string from https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/jetbot/camera.py
    '''
    def gstreamer_pipeline(self, capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21, flip_method=0) :   
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                capture_width, capture_height, framerate, flip_method, output_width, output_height)
    
    def __init__(self, image_w=160, image_h=120, image_d=3, capture_width=3280, capture_height=2464, framerate=60, gstreamer_flip=0):
        '''
        gstreamer_flip = 0 - no flip
        gstreamer_flip = 1 - rotate CCW 90
        gstreamer_flip = 2 - flip vertically
        gstreamer_flip = 3 - rotate CW 90
        '''
        self.w = image_w
        self.h = image_h
        self.running = True
        self.frame = None
        self.flip_method = gstreamer_flip
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.framerate = framerate

    def init_camera(self):
        import cv2

        # initialize the camera and stream
        self.camera = cv2.VideoCapture(
            self.gstreamer_pipeline(
                capture_width =self.capture_width,
                capture_height =self.capture_height,
                output_width=self.w,
                output_height=self.h,
                framerate=self.framerate,
                flip_method=self.flip_method),
            cv2.CAP_GSTREAMER)

        self.poll_camera()
        print('CSICamera loaded.. .warming camera')
        time.sleep(2)
        
    def update(self):
        self.init_camera()
        while self.running:
            self.poll_camera()

    def poll_camera(self):
        import cv2
        self.ret , frame = self.camera.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def run(self):
        self.poll_camera()
        return self.frame

    def run_threaded(self):
        return self.frame
    
    def shutdown(self):
        self.running = False
        print('stoping CSICamera')
        time.sleep(.5)
        del(self.camera)

class V4LCamera(BaseCamera):
    '''
    uses the v4l2capture library from this fork for python3 support: https://github.com/atareao/python3-v4l2capture
    sudo apt-get install libv4l-dev
    cd python3-v4l2capture
    python setup.py build
    pip install -e .
    '''
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20, dev_fn="/dev/video0", fourcc='MJPG'):

        self.running = True
        self.frame = None
        self.image_w = image_w
        self.image_h = image_h
        self.dev_fn = dev_fn
        self.fourcc = fourcc

    def init_video(self):
        import v4l2capture

        self.video = v4l2capture.Video_device(self.dev_fn)

        # Suggest an image size to the device. The device may choose and
        # return another size if it doesn't support the suggested one.
        self.size_x, self.size_y = self.video.set_format(self.image_w, self.image_h, fourcc=self.fourcc)

        print("V4L camera granted %d, %d resolution." % (self.size_x, self.size_y))

        # Create a buffer to store image data in. This must be done before
        # calling 'start' if v4l2capture is compiled with libv4l2. Otherwise
        # raises IOError.
        self.video.create_buffers(30)

        # Send the buffer to the device. Some devices require this to be done
        # before calling 'start'.
        self.video.queue_all_buffers()

        # Start the device. This lights the LED if it's a camera that has one.
        self.video.start()


    def update(self):
        import select
        from donkeycar.parts.image import JpgToImgArr

        self.init_video()
        jpg_conv = JpgToImgArr()

        while self.running:
            # Wait for the device to fill the buffer.
            select.select((self.video,), (), ())
            image_data = self.video.read_and_queue()
            self.frame = jpg_conv.run(image_data)


    def shutdown(self):
        self.running = False
        time.sleep(0.5)



class MockCamera(BaseCamera):
    '''
    Fake camera. Returns only a single static frame
    '''
    def __init__(self, image_w=160, image_h=120, image_d=3, image=None):
        if image is not None:
            self.frame = image
        else:
            self.frame = np.array(Image.new('RGB', (image_w, image_h)))

    def update(self):
        pass

    def shutdown(self):
        pass

class ImageListCamera(BaseCamera):
    '''
    Use the images from a tub as a fake camera output
    '''
    def __init__(self, path_mask='~/mycar/data/**/*.jpg'):
        self.image_filenames = glob.glob(os.path.expanduser(path_mask), recursive=True)
    
        def get_image_index(fnm):
            sl = os.path.basename(fnm).split('_')
            return int(sl[0])

        '''
        I feel like sorting by modified time is almost always
        what you want. but if you tared and moved your data around,
        sometimes it doesn't preserve a nice modified time.
        so, sorting by image index works better, but only with one path.
        '''
        self.image_filenames.sort(key=get_image_index)
        #self.image_filenames.sort(key=os.path.getmtime)
        self.num_images = len(self.image_filenames)
        print('%d images loaded.' % self.num_images)
        print( self.image_filenames[:10])
        self.i_frame = 0
        self.frame = None
        self.update()

    def update(self):
        pass

    def run_threaded(self):        
        if self.num_images > 0:
            self.i_frame = (self.i_frame + 1) % self.num_images
            self.frame = Image.open(self.image_filenames[self.i_frame]) 

        return np.asarray(self.frame)

    def shutdown(self):
        pass
