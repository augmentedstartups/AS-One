import copy
import warnings
import cv2
from loguru import logger
import os
import time
import asone.utils as utils
from asone.trackers import Tracker
from asone.detectors import Detector
from asone.recognizers import TextRecognizer
from asone.segmentors import Segmentor
from asone.utils.default_cfg import config
from asone.utils.video_reader import VideoReader
from asone.utils import compute_color_for_labels

import numpy as np


class ASOne:
    def __init__(self,
                 detector: int = 0,
                 tracker: int = -1,
                 segmentor: int = -1,
                 weights: str = None,
                 segmentor_weights: str = None,
                 use_cuda: bool = True,
                 recognizer: int = None,
                 languages: list = ['en'],
                 num_classes=80
                 ) -> None:

        self.use_cuda = use_cuda
        self.use_segmentation = False
        
        # Check if user want to use segmentor
        if segmentor != -1:
            self.use_segmentation = True

            # Load Segmentation model
            self.segmentor = self.get_segmentor(segmentor, segmentor_weights)

        # get detector object
        self.detector = self.get_detector(detector, weights, recognizer, num_classes)
        self.recognizer = self.get_recognizer(recognizer, languages=languages)
    
        if tracker == -1:
            self.tracker = None
            return
            
        self.tracker = self.get_tracker(tracker)

    def get_detector(self, detector: int, weights: str, recognizer, num_classes):
        detector = Detector(detector, weights=weights,
                            use_cuda=self.use_cuda, recognizer=recognizer, num_classes=num_classes).get_detector()
        return detector

    def get_recognizer(self, recognizer: int, languages):
        if recognizer == None:
            return None
        recognizer = TextRecognizer(recognizer,
                            use_cuda=self.use_cuda, languages=languages).get_recognizer()

        return recognizer

    def get_tracker(self, tracker: int):
        tracker = Tracker(tracker, self.detector,
                          use_cuda=self.use_cuda)
        return tracker
    
    def get_segmentor(self, segmentor, segmentor_weights):
        segmentor = Segmentor(segmentor, segmentor_weights, self.use_cuda)
        return segmentor

    def _update_args(self, kwargs):
        for key, value in kwargs.items():
            if key in config.keys():
                config[key] = value
            else:
                print(f'"{key}" argument not found! valid args: {list(config.keys())}')
                exit()
        return config

    def track_stream(self,
                    stream_url,
                    **kwargs
                    ):

        output_filename = 'result.mp4'
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(stream_url, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def track_video(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def detect_video(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
    
    def detect(self, source, **kwargs)->np.ndarray:
        """ Function to perform detection on an img

        Args:
            source (_type_): if str read the image. if nd.array pass it directly to detect

        Returns:
            _type_: ndarray of detection
        """
        if isinstance(source, str):
            source = cv2.imread(source)
        return self.detector.detect(source, **kwargs)

    def detect_and_track(self, frame, **kwargs):
        if self.tracker:
            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(
                frame, kwargs)
            info = None
        else:
            dets, info = self.detect(source=frame, **kwargs)
            bboxes_xyxy = dets[:, :4]
            scores = dets[:, 4]
            class_ids = dets[:, 5]
            ids = None

        return (bboxes_xyxy, ids, scores, class_ids), info
        
    def detect_text(self, image):
        horizontal_list, _ = self.detector.detect(image)
        if self.recognizer is None:
                raise TypeError("Recognizer can not be None")
            
        return self.recognizer.recognize(image, horizontal_list=horizontal_list,
                            free_list=[])

    def track_webcam(self,
                     cam_id=0,
                     **kwargs):
        output_filename = 'results.mp4'

        kwargs['filename'] = output_filename
        kwargs['fps'] = 29
        config = self._update_args(kwargs)


        for (bbox_details, frame_details) in self._start_tracking(cam_id, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
        
    def _start_tracking(self,
                        stream_path: str,
                        config: dict) -> tuple:

        if not self.tracker:
            warnings.warn(f'No tracker has been selected. Only the detector is operational.')

        fps = config.pop('fps')
        output_dir = config.pop('output_dir')
        filename = config.pop('filename')
        save_result = config.pop('save_result')
        display = config.pop('display')
        draw_trails = config.pop('draw_trails')
        class_names = config.pop('class_names')

        cap = self.read_video(stream_path)
        width, height = cap.frame_size
        frame_count = cap.frame_counts

        if fps is None:
            fps = cap.fps

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

        for frame in cap:
            start_time = time.time()

            im0 = copy.deepcopy(frame)
            
            (bboxes_xyxy, ids, scores, class_ids), _ = self.detect_and_track(frame, **config)

            elapsed_time = time.time() - start_time

            logger.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                                 elapsed_time * 1000))

            if self.recognizer:
                res = self.recognizer.recognize(frame, horizontal_list=bboxes_xyxy,
                            free_list=[])
                im0 = utils.draw_text(im0, res)
            else:
                im0 = self.draw(im0,
                                    (bboxes_xyxy, ids, scores, class_ids),
                                    draw_trails=draw_trails,
                                    class_names=class_names,
                                    display=display)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            
            if self.use_segmentation:
                # Will generate mask using SAM
                masks = self.segmentor.create_mask(np.array(bboxes_xyxy), frame)
                im0 = self.draw_masks(im0, masks)
                bboxes_xyxy = (bboxes_xyxy, masks) 
            # if display:
            #     cv2.imshow(' Sample', im0)
            if save_result:
                video_writer.write(im0)

            frame_id += 1

            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

            # yeild required values in form of (bbox_details, frames_details)
            yield (bboxes_xyxy, ids, scores, class_ids), (im0 if display else frame, frame_id-1, fps)

        tac = time.time()
        print(f'Total Time Taken: {tac - tic:.2f}')

    @staticmethod
    def draw(img, dets, display, **kwargs):
        draw_trails = kwargs.get('draw_trails', False)
        class_names = kwargs.get('class_names', None)
        if isinstance(dets, tuple):
            bboxes_xyxy, ids, scores, class_ids = dets
            if isinstance(bboxes_xyxy, tuple):
                bboxes_xyxy, _ = bboxes_xyxy    
                
        elif isinstance(dets, np.ndarray):
            bboxes_xyxy = dets[:, :4]
            scores = dets[:, 4]
            class_ids = dets[:, 5]
            ids = None
        
        img = utils.draw_boxes(img,
                                bbox_xyxy=bboxes_xyxy,
                                class_ids=class_ids,
                                identities=ids,
                                draw_trails=draw_trails,
                                class_names=class_names)
        
        if display:
            cv2.imshow(' Sample', img)
        
        return img
    
    @staticmethod      
    def draw_masks(img, dets, **kwargs):
        color = [0, 255, 0]
        if isinstance(dets, tuple):
            bboxes_xyxy, ids, scores, class_ids = dets
            if isinstance(bboxes_xyxy, tuple):
                bboxes_xyxy, masks = bboxes_xyxy
        else:
            masks = dets
            class_ids = None    
        masked_image = img.copy()
        for idx in range(len(masks)):
            mask = masks[idx].squeeze()  # Squeeze to remove singleton dimension
            if class_ids is not None:
                color = compute_color_for_labels(int(class_ids[idx]))
            color = np.asarray(color, dtype='uint8')
            mask_color = np.expand_dims(mask, axis=-1) * color  # Apply color to the mask
            # Apply the mask to the image
            masked_image = np.where(mask_color > 0, mask_color, masked_image)

        masked_image = masked_image.astype(np.uint8)
        return cv2.addWeighted(img, 0.5, masked_image, 0.5, 0)
    
    def read_video(self, video_path):
        vid = VideoReader(video_path)
        
        return vid
    
if __name__ == '__main__':
    # asone = ASOne(tracker='norfair')
    asone = ASOne()

    asone.start_tracking('data/sample_videos/video2.mp4',
                         save_result=True, display=False)
