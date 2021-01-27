import cv2

coco_file = r'files\coco.names'
path_conf = r'files\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
path_weight = r'files\frozen_inference_graph.pb'

with open(coco_file, 'rt') as object_name:
    names_array = object_name.read().rstrip('\n').split('\n')

detection_model = cv2.dnn_DetectionModel(path_weight, path_conf)

detection_model.setInputSize((320, 320))
detection_model.setInputScale(1.0 / 127.5)
detection_model.setInputMean((127.5, 127.5, 127.5))
detection_model.setInputSwapRB(True)

capture = cv2.VideoCapture(1)
sfull, proof = capture.read()

try:
    id_obj, configs, box = detection_model.detect(proof, confThreshold=0.45)
except:
    capture = cv2.VideoCapture(0)

capture.set(3, 640)
capture.set(4, 480)


while True:
    sfull, img = capture.read()

    id_obj, configs, box = detection_model.detect(img, confThreshold=0.45)

    if len(id_obj) != 0:
        for id_s, confidence_s, box_s in zip(id_obj.flatten(), configs.flatten(), box):
            cv2.rectangle(img, box_s, color=(0, 255, 0), thickness=2, lineType=-1, shift=None)
            cv2.putText(img, names_array[id_s - 1].upper() + " - To: " + str(round(confidence_s * 100, 2)),
                        (box_s[0] + 10, box_s[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Out", img)
    cv2.waitKey(1)
