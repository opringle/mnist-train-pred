import bottle
import argparse
import logging
import time
import mxnet as mx
from src import model
import numpy as np
import cv2
import os


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

    output = net(data.as_in_context(mx.cpu()))
    sm_output = mx.nd.exp(output) / mx.nd.sum(mx.nd.exp(output))[0]

    pred = int(mx.nd.argmax(sm_output, axis=1).asscalar())
    conf = float(mx.nd.max(sm_output, axis=1).asscalar())

    response = ModelAdapter.response(float(pred), float(conf), time_taken=time.clock() - start_time)
    logging.info('Transform response: %r', response)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start model in a web server. '
    )

    # Computation
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--params-file', type=str, required=True,
                        help='path to saved network parameters')
    group.add_argument('--gpu-pred', action='store_true',
                        help='include to predict on gpu')
    group.add_argument('--no-hybridize', action='store_true',
                        help='use symbolic network graph for increased computational eff')

    # Network
    group = parser.add_argument_group('Network arguments')
    group.add_argument('--host', type=str, default='localhost')
    group.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    logging.info("Loading model")
    net = model.CnnClassifier(dropout=0, num_label=10)
    net.load_parameters(args.params_file)

    logging.info("Starting server")
    bottle.run(host=args.host, port=args.port, debug=True)
