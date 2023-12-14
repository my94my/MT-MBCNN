from argparse import ArgumentParser
from datetime import datetime
from warnings import filterwarnings
from random import randint

from torch import argmax, cat, tensor, mean, stack, exp, sum
from torch.nn.functional import nll_loss, mse_loss
from torch.optim import Adam

from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds
from data_loader_2a_noise import load_gdf
from models.MTMBCNN import MTMBCNN


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    t = sum(pow(x - y, 2), dim=2)
    return t


def center_loss(xpool, ture_label):
    xpool = xpool.detach()
    feature0 = []
    feature1 = []
    for i in range(len(xpool)):
        if ture_label[i]:
            feature0.append(xpool[i])
        else:
            feature1.append(xpool[i])
    feature0 = stack(feature0, dim=0)
    feature1 = stack(feature1, dim=0)
    feature0 = mean(feature0, dim=0)
    feature1 = mean(feature1, dim=0)
    proto_dists = euclidean_dist(feature0.squeeze(1), feature1.squeeze(1))
    proto_dists = exp(-0.5 * proto_dists)
    return sum(proto_dists)


def loss_function(pred_label, ture_label, x_guide1, x_guide2, x_guide3, xpool1_1, xpool2_1, xpool3_1, feature):
    loss_1 = nll_loss(pred_label, ture_label)
    loss_2 = mse_loss(x_guide1, xpool1_1) * 0.1
    loss_3 = mse_loss(x_guide2, xpool2_1) * 0.1
    loss_4 = mse_loss(x_guide3, xpool3_1) * 0.1
    loss_5 = center_loss(feature, ture_label) * 3.33e-4
    return loss_1 + loss_2 + loss_3 + loss_4 + loss_5


def accuracy(model, dataset):
    label_pred = model(dataset.X.cuda())[0]
    label_pred = argmax(label_pred, dim=1).squeeze().cpu()
    return sum(label_pred.eq(dataset.y)).item() / label_pred.__len__()


def train(n_chan, train_set, valid_set, test_set):
    model = MTMBCNN(n_chan)
    model.cuda()

    Experiment(model, train_set, valid_set, test_set,
               loss_function=loss_function,
               iterator=BalancedBatchSizeIterator(batch_size=32),
               optimizer=Adam(model.parameters(), lr=0.001, weight_decay=0.01),
               model_constraint=MaxNormDefaultConstraint(),
               monitors=[LossMonitor(), MisclassMonitor(), RuntimeMonitor()],
               stop_criterion=Or([MaxEpochs(max_epochs=500), NoDecrease("valid_misclass", num_epochs=150)]),
               remember_best_column="valid_misclass",
               run_after_early_stop=False,
               pin_memory=True,
               cuda=True).run()

    return accuracy(model, test_set)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=1)
    args = parser.parse_args()
    filterwarnings('ignore')
    set_random_seeds(seed=datetime.now().microsecond, cuda=True)
    # load data
    data, label, low_hz_noise, high_hz_noise = load_gdf('./dataA', args.subject, low_cut_hz=0, high_cut_hz=100,
                                                        noise_low=40, noise_high=70,
                                                        start_offset_ms=-500, end_offset_ms=4000)
    types = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for s in range(len(types)):
        t_data = []
        t_label = []
        low_hz_noise_0 = []
        low_hz_noise_1 = []
        for i in range(len(label)):
            if label[i] == types[s][0]:
                t_data.append(data[i])
                t_label.append(0)
                low_hz_noise_0.append(low_hz_noise[i])
            elif label[i] == types[s][1]:
                t_data.append(data[i])
                t_label.append(1)
                low_hz_noise_1.append(low_hz_noise[i])
        t_data = tensor(t_data)
        t_label = tensor(t_label)

        # separation
        n_train = 112
        n_test = 144
        test_data = t_data[n_test:]
        test_label = t_label[n_test:]
        valid_data = t_data[n_train: n_test]
        valid_label = t_label[n_train: n_test]
        train_data = t_data[:n_train]
        train_label = t_label[:n_train]

        # argumentation
        aug_data = []
        for _ in range(1):
            for i in range(n_train):
                aug_data.append(train_data[i] - high_hz_noise[i] + high_hz_noise[randint(0, len(high_hz_noise) - 1)])
        for _ in range(2):
            for i in range(n_train):
                if t_label[i]:
                    aug_data.append(
                        train_data[i] - high_hz_noise[i] + high_hz_noise[randint(0, len(high_hz_noise) - 1)] -
                        low_hz_noise_1[i] + low_hz_noise_1[randint(0, len(low_hz_noise_1)//2 - 1)])
                else:
                    aug_data.append(
                        train_data[i] - high_hz_noise[i] + high_hz_noise[randint(0, len(high_hz_noise) - 1)] -
                        low_hz_noise_0[i] + low_hz_noise_0[randint(0, len(low_hz_noise_0)//2 - 1)])
        aug_data = stack(aug_data, dim=0)
        train_data = cat([train_data, aug_data], dim=0)
        train_label = cat([train_label] * 4, dim=0)
        print('Shape:', train_data.shape, train_label.shape, valid_data.shape, valid_label.shape, test_data.shape,
              test_label.shape)

        test_set = SignalAndTarget(test_data, test_label)
        valid_set = SignalAndTarget(valid_data, valid_label)
        train_set = SignalAndTarget(train_data, train_label)

        acc = train(22, train_set, valid_set, test_set)
        with open(f'result.txt', 'a') as file:
            file.write(
                '{' + f'"subject":{args.subject},"type":{types[s]},"accuracy":{acc},"time":"{datetime.now()}"' + '}\n')
