import cv2

img_file = '015cde0e-car-crash-highway--1024x555.jpg'  # never  left space without a cause as in python it will give error
#video=cv2.VideoCapture('videoplayback.mp4')
video = cv2.VideoCapture('videoplayback1.mp4')
#video = cv2.VideoCapture('pedestrian_and_cars.mp4')
# oretrained class classifier
classifier_file_cars = 'cars.xml'
classifier_file_pedestrian = 'pedestraian.xml'

# create cars classifier
car_tracker = cv2.CascadeClassifier(classifier_file_cars)
ped_tracker = cv2.CascadeClassifier(classifier_file_pedestrian)
# run forever until the car crashes
while True:
    (read_successful, frame) = video.read()  # multiple items in a single varieable  also called as tuple
    if read_successful:
        # must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = ped_tracker.detectMultiScale(grayscaled_frame)
    # draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('soubhik', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
    """
#create open cv image
#run forever until car stops or something or crashes

img = cv2.imread(img_file)

#convert to grayscale(need for harcascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#draw rectangle around the cars
for(x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)


#display the image with the faces spotted
cv2.imshow('soubhiks ', img)
#we have to use the wait key
cv2.waitKey()
print("code completed")
"""
