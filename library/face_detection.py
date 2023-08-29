import cv2
from PIL import Image, ImageDraw
import numpy


# def detect(filename, cascade_file="../lbpcascade_animeface.xml"):
#     if not os.path.isfile(cascade_file):
#         raise RuntimeError("%s: not found" % cascade_file)

#     cascade = cv2.CascadeClassifier(cascade_file)
#     image = cv2.imread(filename, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     faces = cascade.detectMultiScale(
#         gray,
#         # detector options
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(24, 24),
#     )
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


class FaceDetection:
    def __init__(self) -> None:
        self.cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")

    def predict(self, image: Image.Image, scale_head=False):
        grayscale = image.convert("L")
        gray = numpy.array(grayscale)
        faces = self.cascade.detectMultiScale(
            gray,
            # detector options
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        draw = ImageDraw.ImageDraw(image)
        faces = []
        for (x, y, w, h) in faces:
            faces.append(
                {
                    "top_left":[x,y],
                    "size": [w,h]
                }
            )
            
            draw.rectangle((x,y,x+(w),y+(h)),outline="BLACK", width=1)
        print(faces)
        image.show()
        return faces
