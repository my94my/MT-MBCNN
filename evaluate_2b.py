from argparse import ArgumentParser
from datetime import datetime
from warnings import filterwarnings

from torch import tensor, load
from braindecode.datautil.signal_target import SignalAndTarget
from data_loader_2b_noise import load_gdf
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
    data, label, _, _ = load_gdf('./dataB', args.subject, low_cut_hz=0, high_cut_hz=100,
                                                        noise_low=40, noise_high=70,
                                                        start_offset_ms=-500, end_offset_ms=4000)
    t_data = tensor(data)
    t_label = tensor(label)

    # separation
    n_test = 340
    test_data = t_data[n_test:680]
    test_label = t_label[n_test:680]

    test_set = SignalAndTarget(test_data, test_label)

    acc = evaluate(f'best_model/2b/{args.subject}.pkl', test_set)
    with open(f'result.txt', 'a') as file:
        file.write(
            '{' + f'"subject":{args.subject},"type":2b,"accuracy":{acc},"time":"{datetime.now()}"' + '}\n')
