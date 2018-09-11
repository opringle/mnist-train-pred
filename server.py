import bottle
import argparse
import logging
import time
import mxnet as mx
from src import model
import numpy as np
import cv2
import os
import multiprocessing


class ModelAdapter(object):

    @staticmethod
    def response(pred, conf, time_taken):
        return {
            'prediction': pred,
            'confidence': conf,
            'time_taken': '{:.4} seconds'.format(time_taken)
        }


@bottle.post('/transform')
def transform():
    start_time = time.clock()

    upload = bottle.request.files.get('image')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
        return 'File extension not allowed.'

    upload.save('tmp.jpg', overwrite=True)
    image = cv2.imread('tmp.jpg', 0)
    if image.shape != (28, 28):
        return 'Image shape not compatible with network'

    data = mx.nd.array(image).reshape((1, 28, 28, 1))
    data = data.astype(np.float32) / 255

    output = net(orig_data.as_in_context(mx.cpu()))
    print("Output layer sums to: {}".format(mx.nd.sum(output)))

    pred = int(mx.nd.argmax(output, axis=1).asscalar())
    conf = float(mx.nd.max(output, axis=1).asscalar())

    response = ModelAdapter.response(float(pred), float(conf), time_taken=time.clock() - start_time)
    logging.info('Transform response: %r', response)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start model in a web server. '
    )

    # Computation
    group = parser.add_argument_group('Computation arguments')
    parser.add_argument('--params-file', type=str, required=True,
                        help='path to saved network parameters')
    parser.add_argument('--gpu-pred', action='store_true',
                        help='include to predict on gpu')
    parser.add_argument('--no-hybridize', action='store_true',
                        help='use symbolic network graph for increased computational eff')

    # Network
    group = parser.add_argument_group('Network arguments')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    logging.info("Loading model")
    net = model.CnnClassifier(dropout=0, num_label=10)
    net.load_parameters(args.params_file)

    # Remove once confidence sums correctly
    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32) / 255, label)),
        batch_size=1,
        num_workers=multiprocessing.cpu_count(),
        shuffle=False)

    for i, (data, label) in enumerate(test_data):
        if i < 1:
            orig_data = data

    logging.info("Starting server")
    bottle.run(host=args.host, port=args.port, debug=True)
