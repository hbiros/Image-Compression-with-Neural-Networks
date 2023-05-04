from typing import Dict

from imageio.core.util import Array, asarray
from tensorflow import Tensor
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.trackable.data_structures import NoDependency
import torch as pt

from .pu_pieapp.models.common import PUPieAPP


class PUPieAppMetric(Metric):
    """
    Class responsible for computing PUPieAPP quality score.
    """

    def __init__(self,
                 name: str = "PU_PieApp score",
                 weights_path: str = "./pupieapp_weights.pt"):
        """
        PUPieAPPMetric class constructor that prepares network that will be used to compute metric and sets up parameters for images when doing so.

        :param name: Metric name.
        :type name: str
        :param weights_path: Path pointing to file with weights that were used to train PUPieAPP network.
        :type weights_path: str
        """
        super().__init__(name=name)
        self._prepare_network(weights_path=weights_path)
        self._set_image_parameters()
        self._metrics = {}

    def update_state(self,
                    #  image_name: str,
                     source_image: Array,
                     compressed_image: Array):
        """
        Method used to update dictionary storing metric values.

        :param image_name: Name of image that will be used to check its PUPieAPP quality score.
        :type image_name: str
        :param source_image: Source image (before compression).
        :type source_image: Array
        :param compressed_image: Compressed image.
        :type compressed_image: Array
        """
        source_image = source_image.numpy()
        compressed_image = compressed_image.numpy()
        source_image_processed = self._process_image(source_image)
        compressed_image_processed = self._process_image(compressed_image)
        with pt.no_grad():
            self._metrics["image_name"] = self._compute_score(source_image_processed=source_image_processed,
                                                            compressed_image_processed=compressed_image_processed)

    def result(self) -> Dict[str, Tensor]:
        """
        Method that returns dictionary holding metric values.

        :return: Dictionary with pairs image_name: metric_value.
        :rtype: Dict[str, Tensor]
        """
        return self._metrics

    def _process_image(self,
                       image: Array) -> pt.Tensor:
        """
        Method used to process loaded image and create pytorch tensor from it.

        :param image: Image loaded by imageio.imread.
        :type image: Array
        :return: Tensor that represents processed image.
        :rtype: pt.Tensor
        """
        return pt.from_numpy(asarray(image)) \
            .permute(2, 0, 1) \
            .unsqueeze(0)

    def _compute_score(self,
                       source_image_processed: pt.Tensor,
                       compressed_image_processed: pt.Tensor) -> Tensor:
        """
        Method used to compute PUPieApp quality score.

        :param source_image_processed: Tensor representing source image (before compression).
        :type source_image_processed: pt.Tensor
        :param compressed_image_processed: Tensor representing compressed image.
        :type compressed_image_processed: pt.Tensor
        :return: PUPieApp quality score.
        :rtype: float
        """
        return self._network(img=source_image_processed,
                             ref=compressed_image_processed,
                             im_type=self._dynamic_range,
                             lum_bottom=self._lum_bottom,
                             lum_top=self._lum_top,
                             stride=self._stride) \
            .numpy()[0][0]

    def _set_image_parameters(self,
                              dynamic_range: str = 'sdr',
                              lum_bottom: float = 0.5,
                              lum_top: float = 100,
                              stride: int = 32):
        """
        Method called by class constructor to set up image parameter values used when evaluating PUPieAPP metric value.

        :param dynamic_range: SDR or HDR.
        :type dynamic_range: str
        :param lum_bottom: Low value used to establish luminance range.
        :type lum_bottom: float
        :param lum_top: High value used to establish luminance range.
        :type lum_top: float
        :param stride: Size of windows used when sampling image patches.
        :type stride: int
        """
        self._dynamic_range = dynamic_range
        self._lum_bottom = lum_bottom
        self._lum_top = lum_top
        self._stride = stride

    def _prepare_network(self,
                         weights_path: str):
        """
        Method called by class constructor to set up network used for evaluating PUPieAPP metric values.

        :param weights_path: Path pointing to file with weights that were used to train PUPieAPP network.
        :type weights_path: str
        """
        if pt.cuda.is_available():
            self._map_location = 'cuda:0'
        else:
            self._map_location = 'cpu'
        self._state = NoDependency(pt.load(weights_path,
                                           map_location=self._map_location))
        self._network = PUPieAPP(state=self._state)
        self._network.eval()
