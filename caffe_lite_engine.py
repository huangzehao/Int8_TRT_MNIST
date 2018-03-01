from __future__ import print_function
import sys
try:
    import tensorrt
    from tensorrt.parsers import caffeparser
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1)

try:
    from PIL import Image
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)
import numpy as np
from random import randint
from LoadMNIST import read
from calibrator import PythonEntropyCalibrator, ImageBatchStream

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

ARGS = PARSER.parse_args()
DATA_DIR = ARGS.datadir


#Get the mean image from the caffe binaryproto file
parser = caffeparser.create_caffe_parser()
mean_blob = parser.parse_binary_proto(DATA_DIR + "/mnist/mnist_mean.binaryproto")
# parser.destroy()
MEAN = mean_blob.get_data(28 * 28)

def sub_mean(img):
    '''
    A function to subtract the mean image from a test case
    Will be registered in the Lite Engine preprocessor table
    to be applied to each input case
    '''
    img = img.ravel()
    data = np.empty(len(img))
    for i in range(len(img)):
        data[i] = np.float32((img[i]) - MEAN[i])
    return data.reshape(1, 28, 28)

#Lamba to apply argmax to each result after inference to get prediction
argmax = lambda res: np.argmax(res.reshape(10))

def generate_cases(dataset, number):
    '''
    Generate a list of data to process and answers to compare to
    '''
    cases = []
    labels = []
    loader = read(dataset,"mnist")
    for label, im in loader:
        arr = 255 - im
        arr = arr.reshape(1, 28, 28)
        cases.append(arr)
        labels.append(label)
    return cases[:number], labels[:number]

def main():
    #Generate cases
    train_data, train_target = generate_cases('training', 50000)
    test_data, test_target = generate_cases('testing', 10000)
    #Calibrator
    batchstream = ImageBatchStream(100, train_data, sub_mean)
    int8_calibrator = PythonEntropyCalibrator(["data"], batchstream)

    mnist_engine = tensorrt.lite.Engine(framework="c1",                              #Source framework
                                    deployfile=DATA_DIR + "/mnist/mnist.prototxt",   #Deploy file
                                    modelfile=DATA_DIR + "/mnist/mnist.caffemodel",  #Model File
                                    max_batch_size=100,                              #Max number of images to be processed at a time
                                    input_nodes={"data":(1,28,28)},                  #Input layers
                                    output_nodes=["prob"],                           #Ouput layers
                                    preprocessors={"data":sub_mean},                 #Preprocessing functions
                                    postprocessors={"prob":argmax},                  #Postprocesssing functions
                                    data_type=tensorrt.infer.DataType.HALF)
    # infer
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int)
    for i in range(0, len(test_target), 100):
        batch_data = test_data[i:i+100]
        batch_target = test_target[i:i+100]
        results = mnist_engine.infer(batch_data)[0]
        #Validate results
        correct = np.sum(batch_target == results).astype(np.float32)
        print("Batch [{}] Inference: {:.2f}% Correct".format(i / 100 + 1, (correct / len(batch_target)) * 100))

if __name__ == "__main__":
    main()
