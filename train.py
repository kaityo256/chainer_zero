import collections
import random

import chainer
from chainer import training
from chainer.training import extensions

from data import Data
from model import Model

epoch = 200
batchsize = 100
gpu = -1


def main(f_on, f_off):
    data = Data.load_data(f_on, f_off)
    random.seed(1)
    np.random.seed(1)
    random.shuffle(data)
    dataset = Data.make_dataset(data)
    m = Model(len(data[0][0]))
    model = m.get_model()

    optimizer = chainer.optimizers.Adam()
    #optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)
    test_ratio = 0.05
    nt = int(len(data)*test_ratio)
    test = dataset[:nt]
    train = dataset[nt:]
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    # Training
    trainer.run()
    m.save('test.model')


if __name__ == '__main__':
    main("on.txt", "off.txt")
