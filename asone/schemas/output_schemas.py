class ModelOutput:
    def __init__(self):
        self.dets = Detections()
        self.info = ImageInfo()

class Detections:
    def __init__(self):
        self.bbox = None
        self.ids = []
        self.score = []
        self.class_ids = []

class ImageInfo:
    def __init__(self):
        self.image = None
        self.frame_no = None
        self.fps = None