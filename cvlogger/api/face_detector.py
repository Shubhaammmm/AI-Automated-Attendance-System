import numpy as np

from api.ie_module import Module  # Importing a custom module
from api.utils import resize_input  # Importing a function for resizing input

from openvino.runtime import PartialShape  # Importing a class for handling partial shapes in OpenVINO


class FaceDetector(Module):  # Defining a class named FaceDetector which inherits from Module
    class Result:  # Defining a nested class named Result
        OUTPUT_SIZE = 7  # Class attribute defining the output size

        def __init__(self, output):  # Constructor method for Result class
            # Extracting attributes from the output
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4]))  # (x, y)
            self.size = np.array((output[5], output[6]))  # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):  # Method for rescaling region of interest (ROI)
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):  # Method for resizing ROI
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):  # Method for clipping ROI
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)

    def __init__(self, core, model, input_size, confidence_threshold=0.5, roi_scale_factor=1.15):
        # Constructor method for FaceDetector class
        super(FaceDetector, self).__init__(core, model, 'Face Detection')  # Calling the constructor of the superclass

        # Checking the number of input and output layers of the model
        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        # Getting the input tensor name
        self.input_tensor_name = self.model.inputs[0].get_any_name()

        # Reshaping the model if input_size is specified
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Both input height and width should be positive for Face Detector reshape")

        # Storing input and output shapes and layouts
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape

        # Checking if the output shape is as expected
        if len(self.output_shape) != 4 or self.output_shape[3] != self.Result.OUTPUT_SIZE:
            pass
            # raise RuntimeError("The model expects output shape with {} outputs".format(self.Result.OUTPUT_SIZE))

        # Checking the validity of confidence_threshold and roi_scale_factor
        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError("Confidence threshold is expected to be in range [0; 1]")
        if roi_scale_factor < 0.0:
            raise ValueError("Expected positive ROI scale factor")

        # Storing confidence_threshold and roi_scale_factor
        self.confidence_threshold = confidence_threshold
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):  # Method for preprocessing input frame
        self.input_size = frame.shape
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):  # Method for asynchronously starting inference on a frame
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):  # Method for enqueuing input to the inference request
        return super(FaceDetector, self).enqueue({self.input_tensor_name: input})

    def postprocess(self):  # Method for postprocessing inference results
        outputs = self.get_outputs()[0]  # Getting the output from the inference request

        results = []  # List to store processed results

        # Looping through each output to process results
        if len(outputs.shape) == 4:
            for output in outputs[0][0]:
                result = FaceDetector.Result(output)
                if result.confidence < self.confidence_threshold:
                    break  # results are sorted by confidence decrease

                result.resize_roi(self.input_size[1], self.input_size[0])
                result.rescale_roi(self.roi_scale_factor)
                result.clip(self.input_size[1], self.input_size[0])
                results.append(result)
        if len(outputs.shape) == 2:
            results.append(outputs)

        return results  # Returning processed results
