from datetime import datetime
from time import sleep

import cv2
import grovepi
import imutils
import pandas as pd
from picamera import PiCamera
from picamera.array import PiRGBArray


def get_sensor_values():
    try:
        light_value = grovepi.analogRead(LIGHT)
        sound_value = grovepi.analogRead(SOUND)
        return {
            'LIGHT': light_value,
            'SOUND': sound_value
        }
    except IOError:
        return {
            'LIGHT': -1,
            'SOUND': -1
        }


def run():
    avg = None
    log = None
    count = 0
    people_count = 0
    frame_count = 0
    csv_date = None

    print('Process started...')
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        timestamp = datetime.now()

        if log is None:
            csv_date = timestamp.strftime('%y-%m-%d')
            log = pd.DataFrame(columns=['datetime', 'light', 'sound', 'motion', 'contour_average', 'image'])

        frame = f.array
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg is None:
            avg = gray.copy().astype("float")
            rawCapture.truncate(0)
            continue

        cv2.accumulateWeighted(gray, avg, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        thresh = cv2.threshold(frame_delta, DELTA_THRESH, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

        for c in contours:
            area = cv2.contourArea(c)
            if MIN_AREA <= area <= MAX_AREA:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                people_count += 1

        if timestamp.strftime('%S') == '00':
            sensors = get_sensor_values()
            img_name = timestamp.strftime('%y-%m-%d-%H-%M-%S.jpg')
            img_path = 'data/images/{}'.format(img_name)
            row = {
                'datetime': timestamp,
                'light': sensors['LIGHT'],
                'sound': sensors['SOUND'],
                'motion': people_count > 0,
                'contour_average': people_count / (frame_count + 1),
                'image': img_name}
            log = log.append(row, ignore_index=True)
            print(row)
            cv2.imwrite(img_path, frame)
            count += 1
            people_count = 0
            frame_count = 0
            sleep(1)

        frame_count += 1

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if count >= 24 * 60:
            log.to_csv(path_or_buf='data/csv/{}.csv'.format(csv_date), date_format='%m/%d/%y %H:%M:%S', index=False)
            count = 0
            log = None
            if timestamp > STOP_TIME:
                rawCapture.truncate(0)
                break
            
        rawCapture.truncate(0)


if __name__ == '__main__':
    STOP_TIME = datetime(2018, 5, 8)

    LIGHT = 1
    SOUND = 0

    DELTA_THRESH = 2
    MIN_AREA = 1500
    MAX_AREA = 40000
    RESOLUTION = (1296, 736)

    FRAMERATE = 16

    camera = PiCamera()
    camera.resolution = RESOLUTION
    camera.vflip = False
    camera.framerate = FRAMERATE
    rawCapture = PiRGBArray(camera, size=RESOLUTION)
    print('Camera warming up...')
    sleep(5)

    grovepi.pinMode(LIGHT, "INPUT")
    grovepi.pinMode(SOUND, "INPUT")

    run()
