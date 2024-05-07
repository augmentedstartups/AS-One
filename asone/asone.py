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
from asone.schemas.output_schemas import ModelOutput

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
        self.model_output = ModelOutput()
        
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
        
        # Emit the warning for DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("track_stream function is deprecated. Kindly use stream_tracker instead", DeprecationWarning)

        output_filename = 'result.mp4'
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(stream_url, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
    
    def stream_tracker(self,
                    stream_url,
                    **kwargs
                    ):

        output_filename = 'result.mp4'
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(stream_url, config):
            # yeild bbox_details, frame_details to main script
            yield self.format_output(bbox_details, frame_details)

    def track_video(self,
                    video_path,
                    **kwargs
                    ):            
           
        # Emit the warning for DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("track_video function is deprecated. Kindly use video_tracker instead", DeprecationWarning)
              
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
    
    def video_tracker(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield self.format_output(bbox_details, frame_details)

    def detect_video(self,
                    video_path,
                    **kwargs
                    ):            
        
        # Emit the warning for DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("detect_video function is deprecated. Kindly use video_detecter instead", DeprecationWarning)
            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
    
    def video_detecter(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield self.format_output(bbox_details, frame_details)
    
    def detect(self, source, **kwargs)->np.ndarray:
        """ Function to perform detection on an img

        Args:
            source (_type_): if str read the image. if nd.array pass it directly to detect

        Returns:
            _type_: ndarray of detection
        """
        # Emit the warning for DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("detect function is deprecated. Kindly use detecter instead", DeprecationWarning)
            
        if isinstance(source, str):
            source = cv2.imread(source)
        return self.detector.detect(source, **kwargs)
    
    def detecter(self, source, **kwargs):
        """ Function to perform detection on an img

        Args:
            source (_type_): if str read the image. if nd.array pass it directly to detect

        Returns:
            _type_: ndarray of detection
        """
        if isinstance(source, str):
            source = cv2.imread(source)
        dets, _ = self.detector.detect(source, **kwargs)
        bboxes_xyxy = dets[:, :4]
        scores = dets[:, 4]
        class_ids = dets[:, 5]
        ids = None
        info = None
        return self.format_output((bboxes_xyxy, ids, scores, class_ids), info)

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
    
    def detect_track_manager(self, frame, **kwargs):
        if self.tracker:
            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(
                frame, kwargs)
            info = None
        else:
            model_output = self.detecter(source=frame, **kwargs)
            
            info = model_output.info            
            bboxes_xyxy = model_output.dets.bbox
            scores = model_output.dets.score
            class_ids = model_output.dets.class_ids
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
        # Emit the warning for DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("track_webcam function is deprecated. Kindly use webcam_tracker instead", DeprecationWarning)
            
        output_filename = 'results.mp4'

        kwargs['filename'] = output_filename
        kwargs['fps'] = 29
        config = self._update_args(kwargs)


        for (bbox_details, frame_details) in self._start_tracking(cam_id, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
    
    def webcam_tracker(self,
                     cam_id=0,
                     **kwargs):
        output_filename = 'results.mp4'

        kwargs['filename'] = output_filename
        kwargs['fps'] = 29
        config = self._update_args(kwargs)


        for (bbox_details, frame_details) in self._start_tracking(cam_id, config):
            # yeild bbox_details, frame_details to main script
            yield self.format_output(bbox_details, frame_details)
        
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
            try:
                (bboxes_xyxy, ids, scores, class_ids), _ = self.detect_track_manager(frame, **config)
            except:
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
                im0 = self.draw((bboxes_xyxy, ids, scores, class_ids),
                                    img=im0,
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
                if len(bboxes_xyxy) > 0: # Check if bounding box is present or not
                    # Will generate mask using SAM
                    masks = self.segmentor.create_mask(np.array(bboxes_xyxy), frame)
                    im0 = self.draw_masks(masks, img=im0, display=display)
                    bboxes_xyxy = (bboxes_xyxy, masks) 
            
            if save_result:
                video_writer.write(im0)

            frame_id += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # yeild required values in form of (bbox_details, frames_details)
            yield (bboxes_xyxy, ids, scores, class_ids), (im0 if display else frame, frame_id-1, fps)

        tac = time.time()
        print(f'Total Time Taken: {tac - tic:.2f}')

    @staticmethod
    def draw(dets, display=False, img=None, **kwargs):            
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
        
        elif isinstance(dets, ModelOutput):
            bboxes_xyxy = dets.dets.bbox
            ids = dets.dets.ids
            score = dets.dets.score
            class_ids = dets.dets.class_ids
            img = dets.info.image if dets.info.image is not None else img
            frame_no = dets.info.frame_no
            fps = dets.info.fps
        
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
    def draw_masks(dets, display, img=None, **kwargs):
        # Check if bounding box are present
        if isinstance(dets, tuple) and len(dets) > 0 and len(dets[0]) == 0:
            return img
        
        elif isinstance(dets, ModelOutput):
            masks = dets.dets.bbox
            ids = dets.dets.ids
            score = dets.dets.score
            class_ids = dets.dets.class_ids
            img = dets.info.image if dets.info.image is not None else img
            frame_no = dets.info.frame_no
            fps = dets.info.fps
            if isinstance(masks, tuple):
                bboxes_xyxy, masks = masks
            if isinstance(masks, np.ndarray):
                return img
        
        elif isinstance(dets, tuple):
            bboxes_xyxy, ids, scores, class_ids = dets
            if isinstance(bboxes_xyxy, tuple):
                bboxes_xyxy, masks = bboxes_xyxy
        else:
            masks = dets
            class_ids = None
                
        color = [0, 255, 0]
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
        img = cv2.addWeighted(img, 0.5, masked_image, 0.5, 0)
        
        if display:
            cv2.imshow(' Sample', img)
        return img

    def read_video(self, video_path):
        vid = VideoReader(video_path)
        
        return vid
    
    def format_output(self, bbox_details, frame_details):

        # Set detections
        self.model_output.dets.bbox = bbox_details[0]
        self.model_output.dets.ids = bbox_details[1]
        self.model_output.dets.score = bbox_details[2]
        self.model_output.dets.class_ids = bbox_details[3]
        if frame_details:
            # Set image info
            self.model_output.info.image = frame_details[0]
            self.model_output.info.frame_no = frame_details[1]
            self.model_output.info.fps = frame_details[2]

        return self.model_output
    
if __name__ == '__main__':
    # asone = ASOne(tracker='norfair')
    asone = ASOne()

    asone.start_tracking('data/sample_videos/video2.mp4',
                         save_result=True, display=False)
