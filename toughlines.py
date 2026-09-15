import cv2
import numpy as np

# Imagine neagră
img = np.zeros((600, 800), dtype=np.uint8)

# Desenăm câteva linii
cv2.line(img, (100, 100), (700, 100), 255, 3)
cv2.line(img, (200, 150), (200, 500), 255, 3)
cv2.line(img, (300, 500), (650, 200), 255, 3)

# Detectăm muchiile
edges = cv2.Canny(img, 50, 150)

# Detectăm liniile
lines = cv2.HoughLines(
    edges,
    1,
    np.pi / 180,
    100
)

# Desenăm liniile detectate
if lines is not None:

    for line in lines:

        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            255,
            2
        )

cv2.imshow("Linii", img)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()