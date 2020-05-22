import cv2
import numpy as np
import random
from tetris import Tetris
from PIL import Image

# This file is for testing the capture card and for gathering data to train the environment

BUF_SIZE = 5 # How many images to capture on each input of keystroke
IM_WIDTH = 160
IM_HEIGHT = 90

TRAIN_ENVIRONMENT = True
env = Tetris(debug=True)

def show_webcam():
    cam = cv2.VideoCapture(0)
    it = 0
    while True:       
        
        # Display to screen (720p)
        if not TRAIN_ENVIRONMENT:
            ret_val, img = cam.read()
            im = cv2.resize(img, (IM_WIDTH,IM_HEIGHT))
            im = Image.fromarray(im)
            im.save('temp.png')
            img = cv2.resize(img, (1280,720))
            cv2.imshow('Switch Display', img)
            val = cv2.waitKey(1)
            if val == 27:
                break  # esc to quit
    
        if TRAIN_ENVIRONMENT:
            buf = []
            val = 0
            for _ in range(BUF_SIZE):
                ret_val, im = cam.read()
                img = cv2.resize(im, (1280,720))
                cv2.imshow('Switch Display', img)
                im = cv2.resize(im, (IM_WIDTH,IM_HEIGHT))
                buf.append(im)
                val = cv2.waitKey(1)

            if val == 27:
                break
            elif val == 48:
                print("- Saving negative feedback")
                for x in buf:
                    im = Image.fromarray(x)
                    im.save('training/tetris/negative/' + str(random.randrange(2147483647)) + '.png')
            elif val == 43:
                print("  Saving neutral feedback")
                for x in buf:
                    im = Image.fromarray(x)
                    im.save('training/tetris/neutral/' + str(random.randrange(2147483647)) + '.png')
        it += 1
    cv2.destroyAllWindows()


def main():
    show_webcam()


if __name__ == '__main__':
    main()
