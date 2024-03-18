import cv2

def detect_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    num_faces = len(faces)
    return num_faces

if __name__ == '__main__':
    image_path = "sample/sample.png"
    num_faces_detected = detect_faces(image_path)
    print(f"Number of faces detected: {num_faces_detected}")
