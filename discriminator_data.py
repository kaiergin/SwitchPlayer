import cv2
from PIL import Image

path = 'training/tetris_discriminator/'

def create_dataset():
    cap = cv2.VideoCapture(path + 'raw_video/video5.mp4')
    ret = True
    count = 0
    while ret:
        # Read in 6 frames, approximately 1 every 1/10 of a second
        ret, frame = cap.read()
        frame = cv2.resize(frame, (160,90))
        im = Image.fromarray(frame)
        im.save(path + '/average_frames/' + str(count) + '.png')
        count += 1

create_dataset()