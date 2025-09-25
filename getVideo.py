import threading
import time

import cv2
from cv2.typing import MatLike
from logger import Logger


class readVideo:
    def __init__(self, src=0, name="Video_1") -> None:
        try:
            self.name = name
            self.src = src
            self.ret = None
            self._frame = None
            self.stopped = False
            self.loggerReport = Logger(name)
            self.loggerReport.logger.info("Init constructor readVideo")
            self.lock = threading.Lock()
        except Exception as e:
            self.loggerReport.logger.error("Error in readVideo: " + str(e))

    def start(self):
        try:
            self.stream = cv2.VideoCapture(self.src)
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:  # Handle cases where FPS might not be available or valid
                self.fps = 30  # Default to a common FPS if not found
                self.loggerReport.logger.warning(
                    f"Could not determine FPS for {self.src}. Defaulting to {self.fps}."
                )
            else:
                self.loggerReport.logger.info(f"Video FPS: {self.fps}")
            time.sleep(1)
            self.ret, self._frame = self.stream.read()
            if self.stream.isOpened():
                self.loggerReport.logger.info("Creating video thread")
                self.cam_thread = threading.Thread(
                    target=self.get, name=self.name, daemon=True
                )
                self.cam_thread.start()
            else:
                self.loggerReport.logger.warning("video not initialized")

        except Exception as e:
            self.loggerReport.logger.error("Error starting video thread: " + str(e))

    def stop(self):
        self.loggerReport.logger.info("Stopping video thread and releasing resources")
        self.stopped = True
        if hasattr(self, "cam_thread") and self.cam_thread.is_alive():
            self.cam_thread.join(timeout=2)  # Wait for the thread to finish
        if hasattr(self, "stream") and self.stream.isOpened():
            self.stream.release()

    def get(self):
        delay_between_frames = 1.0 / self.fps
        self.loggerReport.logger.info(
            "Starting thread, time between frames is %f" % (delay_between_frames)
        )
        while not self.stopped:
            start_time = time.time()
            ret, frame = self.stream.read()
            if not ret:
                break

            # Use the lock to safely update the shared frame.
            with self.lock:
                self.ret = ret
                self._frame = frame

            processing_time = time.time() - start_time
            sleep_time = delay_between_frames - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def frame(self) -> MatLike:
        frame_copy = None
        with self.lock:
            if not self.ret or self._frame is None:
                return None
            frame_copy = self._frame.copy()
        return frame_copy
