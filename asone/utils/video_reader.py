import cv2
from tqdm import tqdm

class VideoReader:
    def __init__(
        self,
        video_path: str):
        """
        A simple wrapper class over OpenCVs cv2.VideoCapture that
        has the ability to return frames in batches as opposed to
        one by one.

        Args:
            video_path: path to the video file
                            
        """
        self.video_path = video_path        
        self.video = cv2.VideoCapture(video_path)

    def __len__(self):
        return self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def frame_counts(self):
        return self.__len__()

    @property
    def fps(self):
        return int(self.video.get(cv2.CAP_PROP_FPS))
    
    @property
    def frame_size(self):
        return (
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    
    def __iter__(self):
        while True:
            success, frame = self.video.read()
            if success:
                yield frame
            else:
                break
    