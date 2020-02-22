import cv2
import numpy as np
import random
from tetris import Tetris
from PIL import Image
import tensorflow as tf

bot = Tetris(False, True)
TRAIN_CRITIC = False

BUF_SIZE = 5 # How many images to capture on each input
IM_WIDTH = 160
IM_HEIGHT = 90


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
            im = cv2.resize(img, (IM_WIDTH,IM_HEIGHT))
            im = Image.fromarray(im)
            im.save('temp.png')
            
            # Current prediction
            if it % 10 == 0:
                img_data = tf.io.read_file('temp.png')
                im = tf.image.decode_png(img_data, channels=3)
                im = tf.reshape(tf.image.convert_image_dtype(im, tf.float64), (1,90,160,3))
                print(np.round(bot.eval_critic(im), 2))
                '''
                val = bot.eval_critic(im)
                if np.argmax(val) == 0:
                    print("Good move")
                elif np.argmax(val) == 1:
                    print("Bad move")
                else:
                    print("neutral")
                '''
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
                    im = cv2.resize(im, (IM_WIDTH,IM_HEIGHT))
                    buf.append(im)
                    val = cv2.waitKey(1)

                if val == 27:
                    break
                '''
                Temporarily removing positive feedback
                elif val == 32:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("+ Giving positive feedback") # space bar
                    #bot.fit_critic(im, np.array([0.99]))
                    for x in buf:
                        im = Image.fromarray(x)
                        im.save('training/tetris/positive/' + str(random.randrange(2147483647)) + '.png')
                '''
                if val == 48:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("- Giving negative feedback") # q
                    #bot.fit_critic(im, np.array([-0.99]))
                    for x in buf:
                        im = Image.fromarray(x)
                        im.save('training/tetris_critic/negative/' + str(random.randrange(2147483647)) + '.png')
                elif val == 43:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("  Neutral feedback")
                    #bot.fit_critic(im, np.array([0.0]))
                    for x in buf:
                        im = Image.fromarray(x)
                        im.save('training/tetris_critic/neutral/' + str(random.randrange(2147483647)) + '.png')

            else:
                val = getch()
                buf = []
                for _ in range(BUF_SIZE):
                    ret_val, im = cam.read()
                    im = cv2.resize(im, (IM_WIDTH,IM_HEIGHT))
                    buf.append(im)

                if val == b'q':
                    break
                elif val == b' ':
                    #print("Current prediction:", bot.eval_critic(im))
                    print("+ Giving positive feedback") # space bar
                    #bot.fit_critic(im, np.array([0.99]))
                    for x in buf:
                        im = Image.fromarray(x)
                        im.save('training/tetris/positive/' + str(random.randrange(2147483647)) + '.png')
                elif val == b'0':
                    #print("Current prediction:", bot.eval_critic(im))
                    print("- Giving negative feedback") # q
                    #bot.fit_critic(im, np.array([-0.99]))
                    for x in buf:
                        im = Image.fromarray(x)
                        im.save('training/tetris/negative/' + str(random.randrange(2147483647)) + '.png')
                else:
                    #print("Current prediction:", bot.eval_critic(im))
                    print("  Neutral feedback")
                    #bot.fit_critic(im, np.array([0.0]))
                    for x in buf:
                        im = Image.fromarray(x)
                        im.save('training/tetris/neutral/' + str(random.randrange(2147483647)) + '.png')
        it += 1
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()
