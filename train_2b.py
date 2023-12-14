from argparse import ArgumentParser
from datetime import datetime
from warnings import filterwarnings
from random import randint

from torch import cat, tensor, stack
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import set_random_seeds
from data_loader_2b_noise import load_gdf
from train_2a import train

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=1)
    args = parser.parse_args()
    filterwarnings('ignore')
    set_random_seeds(seed=datetime.now().microsecond, cuda=True)
    # load data
    data, label, low_hz_noise, high_hz_noise = load_gdf('./dataB', args.subject, low_cut_hz=0, high_cut_hz=100, noise_low=40, noise_high=70,
                               start_offset_ms=-500, end_offset_ms=4000)
    low_hz_noise_0 = []
    low_hz_noise_1 = []
    for i in range(len(label)):
        if label[i] == 0:
            low_hz_noise_0.append(low_hz_noise[i])
        else:
            low_hz_noise_1.append(low_hz_noise[i])
    t_data = tensor(data)
    t_label = tensor(label)

    # separation
    n_train = 272
    n_test = 340
    test_data = t_data[n_test:680]
    test_label = t_label[n_test:680]
    valid_data = t_data[n_train: n_test]
    valid_label = t_label[n_train: n_test]
    train_data = t_data[:n_train]
    train_label = t_label[:n_train]

    # argumentation
    aug_data = []
    for _ in range(1):
        for i in range(n_train):
            aug_data.append(t_data[i] - high_hz_noise[i] + high_hz_noise[randint(0, len(high_hz_noise) - 1)])
    for _ in range(2):
        for i in range(n_train):
            if t_label[i]:
                aug_data.append(t_data[i] - high_hz_noise[i] + high_hz_noise[randint(0, len(high_hz_noise) - 1)] -
                                low_hz_noise[i] + low_hz_noise_1[randint(0, len(low_hz_noise_1)//2 - 1)])
            else:
                aug_data.append(t_data[i] - high_hz_noise[i] + high_hz_noise[randint(0, len(high_hz_noise) - 1)] -
                                low_hz_noise[i] + low_hz_noise_0[randint(0, len(low_hz_noise_0)//2 - 1)])
    aug_data = stack(aug_data, dim=0)
    train_data = cat([train_data, aug_data], dim=0)
    train_label = cat([train_label] * 4, dim=0)
    print('Shape:', train_data.shape, train_label.shape, valid_data.shape, valid_label.shape, test_data.shape,
          test_label.shape)

    test_set = SignalAndTarget(test_data, test_label)
    valid_set = SignalAndTarget(valid_data, valid_label)
    train_set = SignalAndTarget(train_data, train_label)

    acc = train(3, train_set, valid_set, test_set)
    with open(f'result.txt', 'a') as file:
        file.write(
            '{' + f'"subject":{args.subject},"type":2b,"accuracy":{acc},"time":"{datetime.now()}"' + '}\n')
