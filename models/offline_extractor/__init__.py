from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """
    A unified class for RCNN-based extractor and region-based extractor
    """
    def __init__(self, *kw, **kwargs):
        self.model = None
        self.preprocessor = None

    @abstractmethod
    def get_bbox_feature(self, img, bbox_list):
        """
        params:
        @img: torch.Tensor
        @bbox_list: List[np.ndarray]

        return: List[torch.Tensor]
        """
        pass
