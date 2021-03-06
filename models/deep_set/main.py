"""Script to quickly train new models from the command line, development purposes only"""

from trainer import MultiSetTrainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='MultiSetSequence DeepSet-Network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--path_to_data', default='data/multiset.pickle', metavar='', type=str,
                        help='path to the data directory')
    parser.add_argument('-n', '--num_pixels', default=1500, metavar='', type=int, help='number of training examples')
    parser.add_argument('-c', '--max_ccds', default=30, metavar='', type=int,
                        help='Maximum set lengths for individual CCDs')
    parser.add_argument('-mse', '--mse_loss', default=True, dest='mse_loss', action='store_true', help='Use MSE Loss')
    parser.add_argument('-l1', '--l1_loss', dest='mse_loss', action='store_false', help='Use L1 Loss')
    parser.add_argument('-e', '--no_epochs', default=100, metavar='', type=int, help='No of Epochs')
    parser.add_argument('-b', '--batch_size', default=4, metavar='', type=int,
                        help='BatchSize: Currently only supports 1')
    parser.add_argument('-lr', '--learning_rate', default=0.001, metavar='', type=float, help='Learning Rate')
    parser.add_argument('-r', '--reduction', default='sum', metavar='', type=str,
                        help='Reduction to use in permutation invariant layer: Choose from sum, max, min, mean')

    args = vars(parser.parse_args())

    trainer = MultiSetTrainer(num_pixels=args['num_pixels'],
                              path_to_data=args['path_to_data'],
                              max_set_len=args['max_ccds'],
                              MSEloss=args['mse_loss'],
                              no_epochs=args['no_epochs'],
                              batch_size=args['batch_size'],
                              lr=args['learning_rate'],
                              reduction=args['reduction'])
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()
