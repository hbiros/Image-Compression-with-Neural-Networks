import os

from numpy import array, float32
import tensorflow as tf
from tensorflow.python.trackable.data_structures import NoDependency
import torch as pt

from .pu_pieapp.models.common import PUPieAPP


class PUPieAppMetric(tf.keras.metrics.Metric):
    """
    Class responsible for computing PUPieAPP quality score.
    """

    def __init__(self):
        """
        PUPieAPPMetric class constructor that prepares network that will be used to compute metric and sets up 
        parameters for images when doing so.
        """
        super().__init__(name="PU_PieApp score")
        self._weight_path = self._find_weights()
        self._prepare_network()
        self._set_image_parameters()
        self._metric_list = []

    def update_state(self,
                     source_image: tf.Tensor,
                     compressed_image: tf.Tensor,
                     sample_weight = None):
        """
        Method used to update metric state.

        :param source_image: Tensorflow tensor representing source image (before compression).
        :type source_image: tf.Tensor
        :param compressed_image: Tensorflow tensor representing compressed image.
        :type compressed_image: tf.Tensor
        :param sample_weight: Placeholder for sample_weight (required parameter for update_state).
        :type sample_weight: None
        :return: None
        :rtype: None
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
        Method that returns most recently added metric score. In order to get range of values 
        (like list of scores obtained across epochs),  model.history.history['(val_)PU_PieApp score'] 
        should be called.

        :return: Metric value.
        :rtype: float32
        """
        return self._metric_list[-1]

    def _process_image(self,
                       image: array) -> pt.Tensor:
        """
        Method used to process loaded image and create pytorch tensor from it.

        :param image: Numpy array representing input image.
        :type image: numpy.array
        :return: Pytorch tensor representing processed image.
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

        :param source_image_processed: Pytorch tensor representing source image (before compression).
        :type source_image_processed: pt.Tensor
        :param compressed_image_processed: Pytorch tensor representing compressed image.
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
        Method called by class constructor to set up image parameter values used when evaluating 
        PUPieAPP metric value.

        :param dynamic_range: SDR or HDR.
        :type dynamic_range: str
        :param lum_bottom: Low value used to establish luminance range.
        :type lum_bottom: float
        :param lum_top: High value used to establish luminance range.
        :type lum_top: float
        :param stride: Size of windows used when sampling image patches.
        :type stride: int
        :return: None
        :rtype: None
        """
        self._dynamic_range = dynamic_range
        self._lum_bottom = lum_bottom
        self._lum_top = lum_top
        self._stride = stride

    def _prepare_network(self):
        """
        Method called by class constructor to set up network used for evaluating PUPieAPP metric values.

        :return: None
        :rtype: None
        """
        if pt.cuda.is_available():
            self._map_location = 'cuda:0'
        else:
            self._map_location = 'cpu'
        self._state = NoDependency(pt.load(self._weight_path, 
                                           map_location=self._map_location))
        self._network = PUPieAPP(state=self._state)
        self._network.eval()

    def _find_weights(self) -> str:
        """
        Method called by class constructor in order to find path to pytorch weights. 

        :return: Path to pupieapp_weights.pt file.
        :rtype: str
        """
        for root, dirs, files in os.walk('.'):
            if 'pupieapp_weights.pt' in files:
                return root + "\pupieapp_weights.pt"
