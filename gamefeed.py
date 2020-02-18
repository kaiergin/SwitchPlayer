import cv2
import numpy as np
from tetris import Tetris

bot = Tetris(False, True)
TRAIN_CRITIC = True

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    posOn = False
    negOn = False
    posTime = 0
    negTime = 0
    it = 0
    while True:
        ret_val, img = cam.read()
        im = cv2.resize(img, (640,360))
        im = im / 255.0
        im = np.resize(im, (1,640,360,3))
        # Current predicitions
        if it % 10 == 0:
            print(bot.eval_critic(im))
        
        # Display to screen (720p)
        if not TRAIN_CRITIC:
            img = cv2.resize(img, (1280,720))
            cv2.imshow('Switch Display', img)
            val = cv2.waitKey(1)
            if val == 27:
                break  # esc to quit
    
        if TRAIN_CRITIC:
            val = input()
            if val == 'q':
                break
            elif val == '+':
                print("Current prediction:", bot.eval_critic(im))
                print("+ Giving positive feedback") # space bar
                bot.fit_critic(im, np.array([1]))
            elif val == '0':
                print("Current prediction:", bot.eval_critic(im))
                print("- Giving negative feedback") # q
                bot.fit_critic(im, np.array([0]))
            else:
                print("Current prediction:", bot.eval_critic(im))
                print("  Neutral feedback")
                bot.fit_critic(im, np.array([0.5]))

        it += 1
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()
