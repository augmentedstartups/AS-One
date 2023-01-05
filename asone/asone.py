import copy
import cv2
from loguru import logger
import os
import time
import asone.utils as utils
from asone.trackers import Tracker
from asone.detectors import Detector


class ASOne:
    def __init__(self,
                 tracker: int = 0,
                 detector: int = 0,
                 weights: str = None,
                 use_cuda: bool = True) -> None:

        self.use_cuda = use_cuda

        # get detector object
        self.detector = self.get_detector(detector, weights)

        self.tracker = self.get_tracker(tracker)

    def get_detector(self, detector: int, weights: str):
        detector = Detector(detector, weights=weights, use_cuda=self.use_cuda).get_detector()
        return detector

    def get_tracker(self, tracker: int):

        tracker = Tracker(tracker, self.detector,
                          use_cuda=self.use_cuda)
        return tracker

    def track_video(self, 
                    video_path, 
                    output_dir='results',
                    conf_thres = 0.25, 
                    save_result=True, 
                    display=False, 
                    draw_trails=False, 
                    filter_classes=None,
                    class_names=None):
        
        output_filename = os.path.basename(video_path)

        for (bbox_details, frame_details) in self._start_tracking(video_path,
                                                                    output_filename,
                                                                    output_dir=output_dir,
                                                                    conf_thres=conf_thres,
                                                                    save_result=save_result,
                                                                    display=display,
                                                                    draw_trails=draw_trails,
                                                                    filter_classes=filter_classes,
                                                                    class_names=class_names):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def track_webcam(self, 
                     cam_id=0, 
                     output_dir='results',
                     conf_thres = 0.25, 
                     save_result=False, 
                     display=True, 
                     draw_trails=False, 
                     filter_classes=None,
                     class_names=None):

        output_filename = 'results.mp4'

        for (bbox_details, frame_details) in self._start_tracking(cam_id,
                                                                    output_filename,
                                                                    output_dir=output_dir,
                                                                    fps=29,
                                                                    conf_thres=conf_thres,
                                                                    save_result=save_result,
                                                                    display=display,
                                                                    draw_trails=draw_trails,
                                                                    filter_classes=filter_classes,
                                                                    class_names=class_names):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def _start_tracking(self, 
                        stream_path, 
                        filename,  
                        fps=None, 
                        conf_thres=0.25,
                        output_dir='results',
                        save_result=True, 
                        display=False, 
                        draw_trails=False, 
                        filter_classes=None,
                        class_names=None):

        cap = cv2.VideoCapture(stream_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            logger.info(f"video save path is {save_path}")

            video_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        tic = time.time()

        prevTime = 0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            im0 = copy.deepcopy(frame)

            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(
                frame, conf_thres=conf_thres, filter_classes=filter_classes)
            elapsed_time = time.time() - start_time

            logger.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                                 elapsed_time * 1000), )

            im0 = utils.draw_boxes(im0, bboxes_xyxy, class_ids,
                                   identities=ids, draw_trails=draw_trails, class_names=class_names)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            if display:
                cv2.imshow(' Sample', im0)
            if save_result:
                video_writer.write(im0)

            frame_id += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # yeild required values in form of (bbox_details, frames_details)
            yield (bboxes_xyxy, ids, scores, class_ids), (im0 if display else frame, frame_id-1, fps)
            
        tac = time.time()
        print(f'Total Time Taken: {tac - tic:.2f}')

if __name__ == '__main__':
    # asone = ASOne(tracker='norfair')
    asone = ASOne()

    asone.start_tracking('data/sample_videos/video2.mp4',
                        save_result=True, display=False)
