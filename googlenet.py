import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class Inception(nutszebra_chainer.Model):

    def __init__(self, in_channel, conv1x1=64, reduce3x3=96, conv3x3=128, reduce5x5=16, conv5x5=32, pool_proj=32):
        super(Inception, self).__init__()
        modules = []
        modules.append(('conv1x1', L.Convolution2D(in_channel, conv1x1, 1, 1, 0)))
        modules.append(('reduce3x3', L.Convolution2D(in_channel, reduce3x3, 1, 1, 0)))
        modules.append(('conv3x3', L.Convolution2D(reduce3x3, conv3x3, 3, 1, 1)))
        modules.append(('reduce5x5', L.Convolution2D(in_channel, reduce5x5, 1, 1, 0)))
        modules.append(('conv5x5', L.Convolution2D(reduce5x5, conv5x5, 5, 1, 2)))
        modules.append(('pool_proj', L.Convolution2D(in_channel, pool_proj, 1, 1, 0)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        for name, link in self.modules:
            self[name].W.data = self.weight_relu_initialization(link)
            self[name].b.data = self.bias_initialization(link, constant=0)

    def __call__(self, x, train=False):
        a = F.relu(self.conv1x1(x))
        b = F.relu(self.conv3x3(F.relu(self.reduce3x3(x))))
        c = F.relu(self.conv5x5(F.relu(self.reduce5x5(x))))
        d = F.relu(self.pool_proj(F.max_pooling_2d(x, ksize=(3, 3), stride=(1, 1), pad=(1, 1))))
        return F.concat((a, b, c, d), axis=1)

    @staticmethod
    def _conv_count_parameters(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += Inception._conv_count_parameters(link)
        return count


class Googlenet(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(Googlenet, self).__init__()
        modules = []
        modules += [('conv1', L.Convolution2D(3, 64, (7, 7), (2, 2), (3, 3)))]
        modules += [('conv2_1x1', L.Convolution2D(64, 64, (1, 1), (1, 1), (0, 0)))]
        modules += [('conv2_3x3', L.Convolution2D(64, 192, (3, 3), (1, 1), (1, 1)))]
        modules += [('inception3a', Inception(192, 64, 96, 128, 16, 32, 32))]
        modules += [('inception3b', Inception(256, 128, 128, 192, 32, 96, 64))]
        modules += [('inception4a', Inception(480, 192, 96, 208, 16, 48, 64))]
        modules += [('inception4b', Inception(512, 160, 112, 224, 24, 64, 64))]
        modules += [('inception4c', Inception(512, 128, 128, 256, 24, 64, 64))]
        modules += [('inception4d', Inception(512, 112, 144, 288, 32, 64, 64))]
        modules += [('inception4e', Inception(528, 256, 160, 320, 32, 128, 128))]
        modules += [('inception5a', Inception(832, 256, 160, 320, 32, 128, 128))]
        modules += [('inception5b', Inception(832, 384, 192, 384, 48, 128, 128))]
        modules += [('linear', L.Linear(1024, category_num))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'googlenet_{}'.format(category_num)

    def count_parameters(self):
        count = 0
        count += functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        count += functools.reduce(lambda a, b: a * b, self.conv2_1x1.W.data.shape)
        count += functools.reduce(lambda a, b: a * b, self.conv2_3x3.W.data.shape)
        count += self.inception3a.count_parameters()
        count += self.inception3b.count_parameters()
        count += self.inception4a.count_parameters()
        count += self.inception4b.count_parameters()
        count += self.inception4c.count_parameters()
        count += self.inception4d.count_parameters()
        count += self.inception4e.count_parameters()
        count += self.inception5a.count_parameters()
        count += self.inception5b.count_parameters()
        count += functools.reduce(lambda a, b: a * b, self.linear.W.data.shape)
        return count

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        self.conv2_1x1.W.data = self.weight_relu_initialization(self.conv2_1x1)
        self.conv2_1x1.b.data = self.bias_initialization(self.conv2_1x1, constant=0)
        self.conv2_3x3.W.data = self.weight_relu_initialization(self.conv2_3x3)
        self.conv2_3x3.b.data = self.bias_initialization(self.conv2_3x3, constant=0)
        self.inception3a.weight_initialization()
        self.inception3b.weight_initialization()
        self.inception4a.weight_initialization()
        self.inception4b.weight_initialization()
        self.inception4c.weight_initialization()
        self.inception4d.weight_initialization()
        self.inception4e.weight_initialization()
        self.inception5a.weight_initialization()
        self.inception5b.weight_initialization()
        self.linear.W.data = self.weight_relu_initialization(self.linear)
        self.linear.b.data = self.bias_initialization(self.linear, constant=0)

    def __call__(self, x, train=True):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = F.relu(self.conv2_1x1(h))
        h = F.relu(self.conv2_3x3(h))
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.inception3a(h)
        h = self.inception3b(h)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.inception4a(h)
        h = self.inception4b(h)
        h = self.inception4c(h)
        h = self.inception4d(h)
        h = self.inception4e(h)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.inception5a(h)
        h = F.relu(self.inception5b(h))
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        h = F.dropout(h, ratio=0.4, train=train)
        h = self.linear(h)
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
