import cv2

class preprocess():
    def preprocess1(self):
        im = cv2.imread()
        # resize image
        im =cv2.reshape
        # -----------------normalise intensity--------------------
        norm_image = cv2.normalize(im, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
