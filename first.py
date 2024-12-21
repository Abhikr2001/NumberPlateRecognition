import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# to read the image File
image_path = r'E:\projects\number_plate\car-number-plate-500x500.webp'
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not open or read the image file.")
else:
    # resize
    image = imutils.resize(image, width=500)

    cv2.imshow("original image", image)
    cv2.waitKey(0)

    # convert image to grey scale to reduce complexities and also dimensions, canny algo only work in grey scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray Scale image", gray)
    cv2.waitKey(0)

    # NEXT STEP TO REDUCE NOISE
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Smoother Image", gray)
    cv2.waitKey(0)

    # to find edge
    edged = cv2.Canny(gray, 170, 200)
    cv2.imshow("Canny edge", edged)
    cv2.waitKey(0)

    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Canny after countering", image1)
    cv2.waitKey(0)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None

    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("top 30 contours", image2)
    cv2.waitKey(0)

    count = 0
    name = 1

    for i in cnts:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)

        if len(approx) == 4:
            NumberPlateCount = approx
            x, y, w, h = cv2.boundingRect(i)
            crp_img = image[y:y + h, x:x + w]

            # Save the cropped image
            cv2.imwrite(str(name) + '.png', crp_img)
            print(f"Cropped image saved: {str(name)}.png")
            
            # Print the path to the saved cropped image
            crop_img_loc = str(name) + '.png'
            print(f"Path to cropped image: {crop_img_loc}")

            name += 1
            break

    # Check if NumberPlateCount is not None before drawing contours
    if NumberPlateCount is not None:
        cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)
        cv2.imshow("final image", image)
        cv2.waitKey(0)

        # Display the cropped image
        cv2.imshow("cropped image", cv2.imread(crop_img_loc))
        cv2.waitKey(0)

        # Perform OCR on the cropped image
        text = pytesseract.image_to_string(crp_img, lang='eng', config='--psm 6')
        print("OCR Output:")
        print(text)
        cv2.waitKey(0)
    else:
        print("NumberPlateCount is None. Contours not found.")
