import cv2
import easyocr

harcascade = "haar.xml"
output_file_path = "detected_numbers.txt"  # Path to the output file

# Load Haar cascade classifier
plate_cascade = cv2.CascadeClassifier(harcascade)

# Load EasyOCR reader with adjusted parameters
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set video frame width and height
# cap.set(3, 640)  # width
# cap.set(4, 480)  # height

min_area = 500

detected_numbers = []  # List to store detected license plate numbers

while True:
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x:x + w]

            # Perform OCR on the license plate region
            results = reader.readtext(cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB), detail=0)

            # Filter OCR results based on string length (assuming longer string means better confidence)
            filtered_results = [result for result in results if len(result) >= 6]  # Adjust the length as needed

            if filtered_results:
                detected_text = filtered_results[0]  # Extracting the detected text
                print("Detected License Plate: ", detected_text)

                # Append the detected number to the list
                detected_numbers.append(detected_text)
                with open(output_file_path, "w") as file:
                    for number in detected_numbers:
                        file.write(number + "\n")

    # Comment the next two lines to avoid displaying the frames
    cv2.imshow("Result", img)
    # cv2.imshow("ROI", img_roi)

    key = cv2.waitKey(1)

# Release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
