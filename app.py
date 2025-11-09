import cv2
import numpy as np
import mediapipe as mp

#TODO: NEED TO ADD CANVAS RESET FOR EVERY SPECIFIC MOVE!!!!
#TODO: GITHUB PUSH and gitignore vsvs.

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def is_finger_open(finger_tip_y, finger_pip_y):
    return finger_tip_y < finger_pip_y
def is_fist(hand_landmarks):
    # not including thumb since it can move independently
    if not is_finger_open(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y):
        if not is_finger_open(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                              hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y):
            if not is_finger_open(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y):
                if not is_finger_open(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
                    return True
    return False
cap = cv2.VideoCapture(0)
   
# global variablese
drawing_mode_active = False
last_x, last_y = 0, 0
canvas = np.zeros((720, 1280, 3), dtype=np.uint8) 

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.flip(image,1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if canvas.shape[0] != image_height or canvas.shape[1] != image_width:
      canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    mode_text_color = (0, 0, 255) if drawing_mode_active else (0, 255, 0) # red is stands for open, green is closed
    cv2.putText(image, f'Drawing Mode: {"ON" if drawing_mode_active else "OFF"}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_text_color, 2, cv2.LINE_AA)
    
    # for 'fist' drawing mode
    if results.multi_hand_landmarks:
              for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                  # left hand fist activates drawing mode
                  handedness = results.multi_handedness[idx].classification[0].label
                  if handedness == 'Left': # if left hand
                      if is_fist(hand_landmarks):
                          drawing_mode_active = True
                          cv2.putText(image, 'Fist Detected (Left Hand)',
                                      (10, image_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                      else:
                          drawing_mode_active = False 

                  elif handedness == 'Right': 
                      index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                      
                      if index_tip and drawing_mode_active:
                          current_x = int(index_tip.x * image_width)
                          current_y = int(index_tip.y * image_height)
                          
                          if last_x != 0 or last_y != 0:
                              cv2.line(canvas, (last_x, last_y), (current_x, current_y), (0, 255, 0), 5)
                          
                          last_x, last_y = current_x, current_y
                          
                          cv2.circle(image, (current_x, current_y), 8, (0, 255, 0), -1)

                      else:
                          last_x, last_y = 0, 0

    image = cv2.addWeighted(image, 1, canvas, 0.9, 0)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

      # for index finger positioning and text
      for hand_landmarks in results.multi_hand_landmarks:
    
        #getting the landmarks
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP] # middle thingy
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

        if index_tip and index_pip: # means the user drawing "1"
            
            # if the index finger is open tip must be lower than pip
            if is_finger_open(index_tip.y,index_pip.y):
                
                image_height, image_width, _ = image.shape # normalization
                
                x_coord_i = int(index_tip.x * image_width)
                y_coord_i= int(index_tip.y * image_height)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if not drawing_mode_active:
                  cv2.putText(image, 'Elif,',
                            (x_coord_i, y_coord_i + 10), 
                            font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                
        if middle_tip and middle_pip: # means the user drawing "2"
             
            if is_finger_open(middle_tip.y,middle_pip.y):
                
                image_height, image_width, _ = image.shape # normalization
                
                x_coord_m = int(middle_tip.x * image_width)
                y_coord_m = int(middle_tip.y * image_height)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if not drawing_mode_active:
                  cv2.putText(image, 'Emre',
                            (x_coord_m, y_coord_m + 10), 
                            font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        if ring_tip and ring_pip: # means the user drawing "3"
            
            if is_finger_open(ring_tip.y,ring_pip.y):
                
                image_height, image_width, _ = image.shape # normalization
                
                x_coord_r = int(ring_tip.x * image_width)
                y_coord_r = int(ring_tip.y * image_height)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if not drawing_mode_active:
                  cv2.putText(image, 'seni',
                            (x_coord_r, y_coord_r + 10), 
                            font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        if pinky_tip and pinky_pip: # means the user drawing "4"
            
            if is_finger_open(pinky_tip.y,pinky_pip.y):
                
                image_height, image_width, _ = image.shape # normalization
                
                x_coord_p = int(pinky_tip.x * image_width)
                y_coord_p = int(pinky_tip.y * image_height)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if not drawing_mode_active:
                  cv2.putText(image, 'cok',
                            (x_coord_p, y_coord_p + 10), 
                            font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        if thumb_tip and thumb_cmc: # means the user drawing "5"
          # thumb is a exception since it lengths in x axis; see docs for more  
            if is_finger_open(thumb_cmc.x,thumb_tip.x):
                
                image_height, image_width, _ = image.shape # normalization
                
                x_coord_t = int(thumb_tip.x * image_width)
                y_coord_t = int(thumb_tip.y * image_height)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if not drawing_mode_active:
                  cv2.putText(image, 'seviyor <3',
                            (x_coord_t, y_coord_t + 10), 
                            font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
      #for canvas reset it should be rock and roll hand emote
      if is_finger_open(index_tip.y,index_pip.y) and is_finger_open(pinky_tip.y,pinky_pip.y):
        if not is_finger_open(middle_tip.y,middle_pip.y) and not is_finger_open(ring_tip.y,ring_pip.y):
          if handedness == "Right":
              canvas = np.zeros((720, 1280, 3), dtype=np.uint8) 
              last_x,last_y=0,0
              cv2.putText(image, 'CANVAS CLEARED (ROCK ON!)',
                                (10, image_height - 60),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
             
    print('Handedness:', results.multi_handedness)

    
            
         
    cv2.imshow('hand detection demo',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

