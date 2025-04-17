import numpy as np

from api.utils import cut_rois, resize_input  # Importing utility functions
from api.ie_module import Module  # Importing a custom module


class LandmarksDetector(Module):  # Defining a class named LandmarksDetector which inherits from Module
    POINTS_NUMBER = 5  # Class attribute defining the number of points

    def __init__(self, core, model):  # Constructor method for LandmarksDetector class
        super(LandmarksDetector, self).__init__(core, model, 'Landmarks Detection')  # Calling the constructor of the superclass

        # Checking the number of input and output layers of the model
        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        # Getting the input tensor name and shape
        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3

        # Checking the output shape of the model
        output_shape = self.model.outputs[0].shape
        if not np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1], output_shape):
            raise RuntimeError("The model expects output shape {}, got {}".format(
                [1, self.POINTS_NUMBER * 2, 1, 1], output_shape))

    def preprocess(self, frame, rois):  # Method for preprocessing input frame and regions of interest (ROIs)
        # Cutting ROIs from the frame
        inputs = cut_rois(frame, rois)
        # Resizing inputs to match the input shape of the model
        inputs = [resize_input(input, self.input_shape, self.nchw_layout) for input in inputs]
        return inputs  # Returning preprocessed inputs

    def enqueue(self, input):  # Method for enqueuing input to the inference request
        return super(LandmarksDetector, self).enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois):  # Method for asynchronously starting inference on a frame with ROIs
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def postprocess(self):  # Method for postprocessing inference results
        # Reshaping output to separate x and y coordinates
        results = [out.reshape((-1, 2)).astype(np.float64) for out in self.get_outputs()]
        return results  # Returning processed results
