import cv2
import numpy as np
import random
from tetris import Tetris

bot = Tetris(False, True)
TRAIN_CRITIC = True

BUF_SIZE = 30


class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    it = 0
    while True:       
        
        # Display to screen (720p)
        if not TRAIN_CRITIC:
            ret_val, img = cam.read()
            im = cv2.resize(img, (640,360))
            im = im / 255.0
            im = np.resize(im, (1,640,360,3))
            # Current prediction
            if it % 10 == 0:
                print(bot.eval_critic(im))
            img = cv2.resize(img, (1280,720))
            cv2.imshow('Switch Display', img)
            val = cv2.waitKey(1)
            if val == 27:
                break  # esc to quit
    
        if TRAIN_CRITIC:
            RENDER = True
            if RENDER:
                buf = []
                val = 0
                for _ in range(BUF_SIZE):
                    ret_val, im = cam.read()
                    img = cv2.resize(im, (1280,720))
                    cv2.imshow('Switch Display', img)
                    im = cv2.resize(im, (640,360))
                    im = im / 255.0
                    im = np.resize(im, (1,640,360,3))
                    buf.append(im)
                    val = cv2.waitKey(1)

                img = np.concatenate(buf, axis=0)

                if val == 27:
                    break
                elif val == 32:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("+ Giving positive feedback") # space bar
                    #bot.fit_critic(im, np.array([0.99]))
                    np.save('training/tetris/positive/' + str(random.randrange(2147483647)), img)
                elif val == 48:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("- Giving negative feedback") # q
                    #bot.fit_critic(im, np.array([-0.99]))
                    np.save('training/tetris/negative/' + str(random.randrange(2147483647)), img)
                elif val == 43:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("  Neutral feedback")
                    #bot.fit_critic(im, np.array([0.0]))
                    np.save('training/tetris/neutral/' + str(random.randrange(2147483647)), img)

            else:
                val = getch()
                buf = []
                for _ in range(BUF_SIZE):
                    ret_val, im = cam.read()
                    im = cv2.resize(im, (640,360))
                    im = im / 255.0
                    im = np.resize(im, (1,640,360,3))
                    buf.append(im)

                img = np.concatenate(buf, axis=0)

                if val == b'q':
                    break
                elif val == b' ':
                    print("Current prediction:", bot.eval_critic(im))
                    print("+ Giving positive feedback") # space bar
                    #bot.fit_critic(im, np.array([0.99]))
                    np.save('training/tetris/positive/' + str(random.randrange(2147483647)), img)
                elif val == b'0':
                    print("Current prediction:", bot.eval_critic(im))
                    print("- Giving negative feedback") # q
                    #bot.fit_critic(im, np.array([-0.99]))
                    np.save('training/tetris/negative/' + str(random.randrange(2147483647)), img)
                else:
                    print("Current prediction:", bot.eval_critic(im))
                    print("  Neutral feedback")
                    #bot.fit_critic(im, np.array([0.0]))
                    np.save('training/tetris/neutral/' + str(random.randrange(2147483647)), img)
        it += 1
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()
