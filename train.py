import chainer
import numpy as np
from chainer import training
from chainer.training import extensions
import random
import collections
from data import Data
from model import Model

def main(f_on, f_off):
    data = Data.load_data(f_on,f_off)
    random.seed(1)
    np.random.seed(1)
    random.shuffle(data)
    dataset = Data.make_dataset(data)
    epoch = 200
    batchsize = 100
    units = len(data[0][0])
    m = Model(units)
    model = m.get_model()
    gpu = -1

    # for GPU
    if gpu >= 0:
        chainer.cuda.get_device(0).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    #optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)
    test_ratio = 0.05
    nt = int(len(data)*test_ratio)
    test = dataset[:nt]
    train = dataset[nt:]
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model,device=gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    # Training
    trainer.run()
    if gpu >= 0:
        model.to_cpu()
    m.save('test.model')

if __name__ == '__main__':
    main("on.txt","off.txt")

