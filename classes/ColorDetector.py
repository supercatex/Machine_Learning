import cv2 as cv
import numpy as np


class ColorDetector:

    def __init__(self, lower, upper, min_area=100):
        self.lower = lower
        self.upper = upper
        self.min_area = min_area

    def get_mask(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower, self.upper)
        return mask

    def find_contours(self, mask):
        ret = []
        _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > self.min_area:
                ret.append(cnt)
        return ret

    @staticmethod
    def draw_contours(image, contours):
        img = image.copy()
        cv.drawContours(img, contours, -1, (0, 255, 0), 1)
        return img

    @staticmethod
    def draw_boxes(image, contours):
        img = image.copy()
        for cnt in contours:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.rectangle(img, (box[0, 0], box[1, 1]), (box[2, 0], box[3, 1]), (0, 0, 255), 2)
        return img

    @staticmethod
    def counting_shapes(contours, lenval):
        line = 0
        triangle = 0
        square = 0
        circle = 0
        for cnt in contours:
            arc_length = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, lenval * arc_length, True)
            if len(approx) == 3:
                if triangle >= 6:
                    triangle = 0
                else:
                    triangle = triangle + 1
            elif len(approx) == 4:
                if square >= 6:
                    square = 0
                else:
                    square = square + 1
            elif (len(approx) > 7) and (len(approx) < 9):
                if circle >= 6:
                    circle = 0
                else:
                    circle = circle + 1
            else:
                line = line + 1
        return {"line": line, "triangle": triangle, "square": square, "circle": circle}

    def detect_shapes(self, image, lenval=0.03):
        mask = self.get_mask(image)
        contours = self.find_contours(mask)
        shapes = ColorDetector.counting_shapes(contours, lenval)
        return shapes, contours

    @staticmethod
    def export(shapes):
        img = np.zeros((297, 210, 3), np.uint8)
        cv.circle(img, (150, 40), 20, (0, 0, 255), -1)
        triangle_cnt = np.array([(150, 90), (130, 120), (170, 120)])
        cv.line(img, (130, 180), (170, 180), (0, 0, 255), 4)
        cv.rectangle(img, (170, 230), (130, 270), (0, 0, 255), -1)
        cv.drawContours(img, [triangle_cnt], 0, (0, 0, 255), -1)
        cv.putText(img, str(shapes["circle"]), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(img, str(shapes["triangle"]), (50, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(img, str(shapes["line"]), (50, 190), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(img, str(shapes["square"]), (50, 260), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imwrite('ROV0.png', img)

