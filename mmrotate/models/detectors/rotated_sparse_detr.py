from ..builder import ROTATED_DETECTORS
from .rotated_detr import RotatedDETR


@ROTATED_DETECTORS.register_module()
class RotatedSparseDETR(RotatedDETR):

    def __init__(self, *args, **kwargs):
        super(RotatedDETR, self).__init__(*args, **kwargs)
