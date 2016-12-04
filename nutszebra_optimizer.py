import chainer
from chainer import optimizers
import nutszebra_basic_print


class Optimizer(object):

    def __init__(self, model=None):
        self.model = model
        self.optimizer = None

    def __call__(self, i):
        pass

    def update(self):
        self.optimizer.update()


class OptimizerResnet(Optimizer):

    def __init__(self, model=None, schedule=(int(32000. / (50000. / 128)), int(48000. / (50000. / 128))), lr=0.1, momentum=0.9, weight_decay=1.0e-4, warm_up_lr=0.01):
        super(OptimizerResnet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(warm_up_lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.warmup_lr = warm_up_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i == 1:
            lr = self.lr
            print('finishded warming up')
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerDense(Optimizer):

    def __init__(self, model=None, schedule=(150, 225), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerDense, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerWideRes(Optimizer):

    def __init__(self, model=None, schedule=(60, 120, 160), lr=0.1, momentum=0.9, weight_decay=5.0e-4):
        super(OptimizerWideRes, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr * 0.2
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerSwapout(Optimizer):

    def __init__(self, model=None, schedule=(196, 224), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerSwapout, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerXception(Optimizer):

    def __init__(self, model=None, lr=0.045, momentum=0.9, weight_decay=1.0e-5, period=2):
        super(OptimizerXception, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.period = int(period)

    def __call__(self, i):
        if i % self.period == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerVGG(Optimizer):

    def __init__(self, model=None, lr=0.01, momentum=0.9, weight_decay=5.0e-4):
        super(OptimizerVGG, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        # 150 epoch means (0.94 ** 75) * lr
        # if lr is 0.01, then (0.94 ** 75) * 0.01 is 0.0001 at the end
        if i % 2 == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerGooglenet(Optimizer):

    def __init__(self, model=None, lr=0.0015, momentum=0.9, weight_decay=2.0e-4):
        super(OptimizerGooglenet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i % 8 == 0:
            lr = self.optimizer.lr * 0.96
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerNetworkInNetwork(Optimizer):

    def __init__(self, model=None, lr=0.1, momentum=0.9, weight_decay=1.0e-4, schedule=(int(1.0e5 / (50000. / 128)), )):
        super(OptimizerNetworkInNetwork, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.schedule = schedule

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
