import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Model(object):
    def __init__(self, n_in):
        self.model = L.Classifier(MLP(n_in, 2))

    def load(self, filename):
        chainer.serializers.load_npz(filename, self.model)

    def save(self, filename):
        chainer.serializers.save_npz(filename, self.model)

    def predictor(self, x):
        return self.model.predictor(x)

    def get_model(self):
        return self.model
