from argparse import ArgumentParser
from datetime import datetime
from warnings import filterwarnings

from torch import tensor, load
from braindecode.datautil.signal_target import SignalAndTarget
from data_loader_2a_noise import load_gdf
from train_2a import accuracy


def evaluate(model_path, test_set):
    model = load(model_path)
    model.cuda()
    model.eval()
    return accuracy(model, test_set)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=1)
    args = parser.parse_args()
    filterwarnings('ignore')
    # load data
    data, label, _, _ = load_gdf('./dataA', args.subject, low_cut_hz=0, high_cut_hz=100,
                                                        noise_low=40, noise_high=70,
                                                        start_offset_ms=-500, end_offset_ms=4000)
    types = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for s in range(len(types)):
        t_data = []
        t_label = []
        for i in range(len(label)):
            if label[i] == types[s][0]:
                t_data.append(data[i])
                t_label.append(0)
            elif label[i] == types[s][1]:
                t_data.append(data[i])
                t_label.append(1)
        t_data = tensor(t_data)
        t_label = tensor(t_label)

        # separation
        n_test = 144
        test_data = t_data[n_test:]
        test_label = t_label[n_test:]
        print('Shape:', test_data.shape, test_label.shape)

        test_set = SignalAndTarget(test_data, test_label)

        acc = evaluate(f'best_model/2a/{types[s][0]}{types[s][1]}/{args.subject}.pkl', test_set)
        with open(f'result.txt', 'a') as file:
            file.write(
                '{' + f'"subject":{args.subject},"type":{types[s]},"accuracy":{acc},"time":"{datetime.now()}"' + '}\n')
