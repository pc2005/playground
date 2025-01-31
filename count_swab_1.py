import cv2
import numpy as np

def count_swabs(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_with_circles = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=4,
        param1=10,
        param2=16,
        minRadius=12,
        maxRadius=22
    )

    # If circles are found, draw them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Image with Circles", image_with_circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the count of detected circles (swab tips)
    return circles.shape[1] if circles is not None else 0

if __name__ == "__main__":
    image_path = "./swabs_1.jpg"
    swab_count = count_swabs(image_path)
    print(f"Number of swabs in the image: {swab_count}")