import cv2, time, numpy as np
from ultralytics import YOLO  # type:ignore
from pathlib import Path
from config import constants as c


root = Path.cwd()

model_path = f"{root}/models/yolo11n.pt"


cam = cv2.VideoCapture(c.RAW_VIDEO_PATH)  #type:ignore
print(f"Cam is opened: {cam.isOpened()}")

model = YOLO(c.YOLO11N_PATH, task = "detect", verbose = False)


# counter variables before loops
ids_entered = set()
ids_exited = set()

# This is for the loop / watching the video
ff = c.FRAME_FACTOR
counter = ff
while True:


    ret, frame = cam.read()
    if not ret:
        print("Can't get frame")
        break

    # take 1 frame every n frames, n = c.FRAME_FACTORS    
    counter -= 1
    if counter == -1:
        counter = ff
    if counter != ff:
        continue
    
    # undistort the frame    
    h, w = frame.shape[:2]
    f = w
    cam_matrix = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ])
    distortion_coefficients = np.array([-0.35, 0.05, 0.0, 0.0, 0.0], dtype = np.float32).reshape(1,5)
    frame = cv2.undistort(frame, cam_matrix, distortion_coefficients)
    
    # infer 
    results = model.track(frame,classes = [0], imgsz = c.HALF_IMAGE_SIZE, verbose= False, tracker= "botsort.yaml", persist= True)
    result = list(results)[0]

    # draw the boxes
    frame = result.plot()

    # get the zone(reshaped), and draw it with blending
    zone = np.array(c.POLY, dtype= np.int32).reshape((-1,1,2))
    overlay = frame.copy()
    cv2.fillPoly(img= overlay, pts= [zone], color= c.RED)
    cv2.addWeighted(overlay,float(c.BLENDING_FACTOR), frame, float(1-c.BLENDING_FACTOR), 0,frame) 

    # get the boxes -> calculate footpoint
    if result.boxes and result.boxes.is_track:
        ids_in = set()    
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            cv2.circle(frame, (foot_x, foot_y), c.LINES_THICKNESS * 3, c.WHITE, -1)

            id = box.id.cpu().int().numpy()[0]  #type:ignore
            in_zone = cv2.pointPolygonTest(zone,(foot_x,foot_y),False) > 0
            # store the id of any object having entered the zone
            if in_zone and not id in ids_entered:
                ids_entered.add(id)
                print(f"Object of id {id} just entered the zone")
            if in_zone and not id in ids_in:
                ids_in.add(id)

            
        
        for id in ids_entered:
            if not id in ids_in and not id in ids_exited:
                ids_exited.add(id)
                print(f"Object of id {id} just exited the zone")

    
       
    

    #show annotated frame
    cv2.imshow(c.WINDOW_TITLE, frame)
    cv2.waitKey(1)
    
    # check if X is clicked
    if cv2.getWindowProperty(c.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break



#This is for getting one single frame to do the clicker
'''
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x: {x}, y: {y}")


ret, raw_frame = cam.read()
if not ret:
    print("couldn't get frame")
else: 
    print("could get frame")

h, w = raw_frame.shape[:2]
print(f"height: {h} & width: {w}")
f = w
mtx = np.array([
    [f, 0, w/2],
    [0, f, h/2],
    [0, 0, 1]
])
dist = np.array([-0.35, 0.05, 0.0, 0.0, 0.0], dtype = np.float32).reshape(1,5)
frame = cv2.undistort(raw_frame, mtx, dist)

time.sleep(2)

overlay = frame.copy()

poly_pts = np.array(c.POLY, dtype= np.int32).reshape((-1,1,2))

cv2.fillPoly(img= overlay, pts= [poly_pts], color= c.RED)
print(type(c.BLENDING_FACTOR))                          #type: ignore
cv2.addWeighted(overlay,float(c.BLENDING_FACTOR), frame, float(1-c.BLENDING_FACTOR), 0,frame)

WINDOW_TITLE = "Zone Detector"
cv2.namedWindow(WINDOW_TITLE)
cv2.setMouseCallback(WINDOW_TITLE, mouse_callback)

cv2.imshow(WINDOW_TITLE,frame)
cv2.waitKey(0)
'''
print(f"All of the ids that have entered the zone are: {ids_entered}")

cam.release()
cv2.destroyAllWindows()









''' 
DONE:
finish flags of prepare_model
run_test prepare_model
watch videos -> pick videos, specify the zones to detect =>> source.mp4
make a constant RAW_VIDEO_PATH
undistort the cam with acceptatble 
make a point finder
get the footpoint
change to track: when a person goes into the zone, ID them, track their time in the zone, track when they leave the zone 
write the function for checking footpoint in zone

TODO:
solve the problem of losing tracked IDs
change to deepsort tracker

make zone_detector.py? (ZoneDetector class that wraps around the model)
Implement ZoneDetector methods - there must be alert() method? 
'''
