from numpy import array, float32
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
        self._metric_list = []

    def update_state(self,
                     source_image: Tensor,
                     compressed_image: Tensor,
                     sample_weight = None):
        """
        Method used to update metric state.

        :param source_image: Tensor representing source image (before compression).
        :type source_image: tf.Tensor
        :param compressed_image: Tensor representing compressed image.
        :type compressed_image: tf.Tensor
        :param sample_weight: Placeholder for sample_weight (required parameter for update_state).
        :type sample_weight: None
        """
        source_images = [array(image) for image in source_image.numpy().tolist()]
        compressed_images = [array(image) for image in compressed_image.numpy().tolist()]
        results = []
        for i in range(len(source_images)):
            source_image_processed = self._process_image(source_images[i])
            compressed_image_processed = self._process_image(compressed_images[i])
            with pt.no_grad():
                results.append(self._compute_score(source_image_processed=source_image_processed,
                                                   compressed_image_processed=compressed_image_processed))
        average_pupieapp_score = sum(results)/len(results)
        self._metric_list.append(average_pupieapp_score)

    def result(self) -> float32:
        """
        Method that returns dictionary holding metric values.

        :return: Metric value.
        :rtype: float32
        """
        return self._metric_list

    def _process_image(self,
                       image: array) -> pt.Tensor:
        """
        Method used to process loaded image and create pytorch tensor from it.

        :param image: Numpy array representing input image.
        :type image: numpy.array
        :return: Tensor that represents processed image.
        :rtype: pt.Tensor
        """
        return pt.from_numpy(image) \
            .permute(2, 0, 1) \
            .unsqueeze(0)

    def _compute_score(self,
                       source_image_processed: pt.Tensor,
                       compressed_image_processed: pt.Tensor) -> float32:
        """
        Method used to compute PUPieApp quality score.

        :param source_image_processed: Tensor representing source image (before compression).
        :type source_image_processed: pt.Tensor
        :param compressed_image_processed: Tensor representing compressed image.
        :type compressed_image_processed: pt.Tensor
        :return: PUPieApp quality score.
        :rtype: numpy.float32
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
