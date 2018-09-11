from mxnet import nd, gluon


class ConvPoolBlock(gluon.nn.HybridSequential):
    def __init__(self, num_filters, downsample=True):
        """
        :param num_filters: number of filters in convolutional layer
        :param downsample: maxpool after activation is applied
        """
        super().__init__()
        with self.name_scope():
            self.add(gluon.nn.Conv2D(channels=num_filters, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)))
            # self.add(gluon.nn.BatchNorm(axis=1))
            self.add(gluon.nn.Activation(activation='relu'))
            if downsample:
                self.add(gluon.nn.MaxPool2D(pool_size=(2, 2)))


class ConvLayers(gluon.nn.HybridBlock):
    def __init__(self):
        """
        multiple convolutional blocks as described in task
        """
        super().__init__()
        with self.name_scope():
            self.conv1 = ConvPoolBlock(num_filters=32, downsample=False)
            self.conv2_1 = ConvPoolBlock(num_filters=64)
            self.conv2_2 = ConvPoolBlock(num_filters=64)
            self.conv3_1 = ConvPoolBlock(num_filters=256)
            self.conv3_2 = ConvPoolBlock(num_filters=256)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        :param x: mxnet ndarray of data
        :return: mxnet ndarray of data
        """
        x = x.transpose((0, 3, 2, 1))
        conv1_output = self.conv1(x)
        conv2_1_output = self.conv2_1(conv1_output)
        conv2_2_output = self.conv2_2(conv1_output)

        conv3_1_output = self.conv3_1(conv2_1_output)
        conv3_2_output = self.conv3_2(conv2_2_output)
        return F.concat(*[conv3_1_output, conv3_2_output], dim=1).flatten()


class CnnClassifier(gluon.nn.HybridSequential):
    """
    Convnet for mnist image classification
    """
    def __init__(self, dropout, num_label):
        """
        :param dropout: dropout probability for output from final conv layer
        :param num_label: number of neurons in final network layer
        """
        super().__init__()
        with self.name_scope():
            self.add(ConvLayers())
            self.add(gluon.nn.Dense(units=1000, activation='relu'))
            self.add(gluon.nn.Dropout(rate=dropout))
            self.add(gluon.nn.Dense(units=500, activation='relu'))
            self.add(gluon.nn.Dropout(rate=dropout))
            self.add(gluon.nn.Dense(units=num_label))


if __name__ == "__main__":
    """
    Run unit-test
    """
    x = nd.random.uniform(shape=(32, 1, 28, 28))
    block = ConvPoolBlock(num_filters=10)
    block.initialize()
    y = block(x)
    assert y.shape == (32, 10, 14, 14)
    nd.waitall()
    print("Conv Block Unit-test success!")

    x = nd.random.uniform(shape=(32, 28, 28, 1))
    block = ConvLayers()
    block.initialize()
    y = block(x)
    assert y.shape == (32, 512 * 7 * 7)
    nd.waitall()
    print("Conv Layers Unit-test success!")

    block = CnnClassifier(dropout=0, num_label=10)
    block.initialize()
    y = block(x)
    assert y.shape == (32, 10)
    nd.waitall()
    print("Network Unit-test success!")
