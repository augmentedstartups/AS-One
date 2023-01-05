from asone.trackers import ByteTrack
from asone.trackers import NorFair
from asone.trackers import DeepSort

class Tracker:
    def __init__(self, tracker: int, detector: object, use_cuda=True) -> None:
        
        self.trackers = {
            '0': ByteTrack,
            '1': DeepSort,
            '2': NorFair
        }

        self.tracker = self._select_tracker(tracker, detector, use_cuda=use_cuda)

    def _select_tracker(self, tracker, detector, use_cuda):
        _tracker = self.trackers.get(str(tracker), None)

        if _tracker is not None:
            if _tracker is DeepSort:
                return _tracker(detector, use_cuda=use_cuda)
            else:
                return _tracker(detector)
        else:
            raise ValueError(f'Invalid tracker: {tracker}')

    def detect_and_track(self, image, conf_thres=0.25,  filter_classes=None):
        return self.tracker.detect_and_track(image, conf_thres = conf_thres, filter_classes=filter_classes)

    def get_tracker(self):
        return self.tracker