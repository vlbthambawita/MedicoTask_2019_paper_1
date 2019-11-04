import os
import cv2
import numpy as np
from tqdm import tqdm



video_folder = "/media/vajira/OS/data/sperm_original_working_videos"
output_folder_dense_flow = "/media/vajira/OS/data/generated_250_outputs_original_2nd_frame_stride_10"
output_folder_origina_2nd_frame = "/media/vajira/OS/data/generated_250_outputs_stride_10"


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    print(y.shape)
    fx, fy = flow[int(y[0]), int(x[0])].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return  vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 50, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def run(num_of_frames_to_extract):
    video_list = os.listdir(video_folder)

    for video in video_list:
        video_file = os.path.join(video_folder, video)

        output_path_opt = os.path.join(output_folder_dense_flow, video)
        output_path_org = os.path.join(output_folder_origina_2nd_frame, video)

        if not os.path.exists(output_path_opt):
            os.mkdir(output_path_opt)
        if not os.path.exists(output_path_org):
            os.mkdir(output_path_org)



        print(output_path_opt)

        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("length:", length)

        #ret, frame0 = cap.read() # skip one frame for safety
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        start_point = 1
        stride = 10
        skip_number = length / num_of_frames_to_extract

        print("Skip factor=", skip_number)

        for i in tqdm(range(num_of_frames_to_extract)): # range(length -1)
            outfile_odf = os.path.join(output_path_opt, str(i) + ".png")

            outfile_org = os.path.join(output_path_org, str(i) + ".png")


            cap.set(1, start_point)

            ret, frame1 = cap.read()
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            cap.set(1, start_point + stride)
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
           # prev =
            #frame2 = cap.set(1, start_point + 1)
           # prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)


            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            #

            cv2.imshow('frame2',draw_hsv(flow))
            cv2.imshow('frame', frame2)
            #cv2.imshow('frame3', frame2)
            # cv2.imwrite('opticalfb.png', frame2)
            # cv2.imwrite('opticalhsv.png', rgb)
            cv2.imwrite(outfile_odf, draw_hsv(flow))
            cv2.imwrite(outfile_org, frame2)


            k = cv2.waitKey(1) & 0xff
           #  a = input("Test")
            if k == 27:
                break
            #else :
             #   cv2.imwrite('opticalfb.png',frame2)
            #    cv2.imwrite('opticalhsv.png',rgb)
            #prvs = next
            start_point = start_point + skip_number
            # print("Frame NO:", i)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(250)