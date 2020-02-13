import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        img = cv2.resize(img, (640,360))
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('switch display', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()
