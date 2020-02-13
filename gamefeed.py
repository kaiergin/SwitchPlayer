import cv2
import numpy as np
from tetris import Tetris

bot = Tetris(False, True)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    it = 0
    while True:
        ret_val, img = cam.read()
        img = cv2.resize(img, (640,360))
        im = img / 255.0
        im = np.resize(im, (1,640,360,3))
        if it % 10 == 0:
            print(bot.eval_critic(im))
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('switch display', img)
        it += 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()
