import os
import cv2
import numpy as np
from tqdm import tqdm



video_folder = "/media/vajira/Transcend/datasets/visem-dataset/videos"
output_folder = "/media/vajira/Transcend/datasets/visem-dataset/generated_outputs_optical_flow"

video_list = os.listdir(video_folder)

for video in video_list:
    video_file = os.path.join(video_folder, video)

    output_path = os.path.join(output_folder, video )
    os.mkdir(output_path)
    print(output_path)

    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("length:", length)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    for i in tqdm(range(1000)): # range(length -1)
        outfile = os.path.join(output_path, str(i) + ".png")

        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        #cv2.imshow('frame2',rgb)
        #cv2.imshow('frame3', frame2)
        # cv2.imwrite('opticalfb.png', frame2)
        # cv2.imwrite('opticalhsv.png', rgb)
        cv2.imwrite(outfile, rgb)


        #k = cv2.waitKey(1) & 0xff
       #  a = input("Test")
       # if k == 27:
        #    break
        #else :
         #   cv2.imwrite('opticalfb.png',frame2)
        #    cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
        # print("Frame NO:", i)

cap.release()
cv2.destroyAllWindows()