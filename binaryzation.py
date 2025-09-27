import cv2
import numpy as np
from getVideo import readVideo


def binary(image, method=0, Umin=0, Umax=255) -> np.ndarray:
    if method == 0:
        ret, imgBinary = cv2.threshold(image, Umin, Umax, cv2.THRESH_BINARY)
    else:
        imgBinary = cv2.inRange(image, Umin, Umax)
    return imgBinary.astype(np.uint8)


def binaryHSV(
    image, minH=0, maxH=255, minS=0, maxS=255, minV=0, maxV=255
) -> np.ndarray:
    imgBinary = cv2.inRange(image, (minH, minS, minV), (maxH, maxS, maxV))
    return imgBinary


def frame_contours(frame: np.ndarray, umin: int, umax: int):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_bin = binary(frame_gray, method=0, Umin=umin, Umax=umax)
    # frame_bin_inv = cv2.bitwise_not(frame_bin)

    # frame_filtered = filter_frame(frame_bin)

    contours, _ = cv2.findContours(frame_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def filter_frame(frame_bin):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask_open = cv2.morphologyEx(frame_bin, cv2.MORPH_OPEN, kernel)
    return mask_open


def create_trackbar():
    cv2.namedWindow("Placa")
    cv2.createTrackbar("min", "Placa", 25, 255, lambda x: None)
    cv2.createTrackbar("max", "Placa", 80, 255, lambda x: None)


def normalize_frame(frame: np.ndarray, desired_light: int = 90) -> np.ndarray:
    avg = np.average(frame)
    frame_normalized = frame * (desired_light / avg)
    return frame_normalized


def main() -> None:
    output_video_path = "binary_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para el video de salida
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (320, 200), isColor=False)
    video_source = "./predicted_video.mp4"
    video = readVideo(src=video_source)
    video.start()
    frame = cv2.imread("./placa.png")
    cv2.imshow("Placa", cv2.resize(frame, (320, 200)))
    create_trackbar()
    while True:
        frame = video.frame
        if frame is not None:
            min = cv2.getTrackbarPos("min", "Placa")
            max = cv2.getTrackbarPos("max", "Placa")
            # contours = frame_contours(frame, 20, 255)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_normalized = normalize_frame(frame_gray, 95)
            frame_bin = binary(frame_normalized, method=1, Umin=min, Umax=max)
            # for cnt in contours: x, y, w, h = cv2.boundingRect(cnt) cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.imshow("Placa", cv2.resize(frame_bin, (320, 200)))
            out.write(cv2.resize(frame_bin, (320, 200)))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    video.stop()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video procesado y guardado en {output_video_path}")


if __name__ == "__main__":
    main()
