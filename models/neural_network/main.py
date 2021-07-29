

from trainer import BaseNNTrainer
import argparse

def main():

    parser = argparse.ArgumentParser(description='MultiSetSequence DeepSet-Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '-n', '--num_pixels', default=15000,metavar='', type=int,help='number of training examples' )
    parser.add_argument( '-mse','--mse_loss', default=True, dest='mse_loss',action='store_true', help='Use MSE Loss')
    parser.add_argument( '-l1','--l1_loss', dest='mse_loss',action='store_false', help='Use L1 Loss')
    parser.add_argument( '-e','--no_epochs', default=100,metavar='', type=int,help='No of Epochs')
    parser.add_argument( '-b','--batch_size', default=16,metavar='', type=int,help='BatchSize: Currently only supports 1')
    parser.add_argument( '-lr','--learning_rate', default=0.001,metavar='',type=float, help='Learning Rate')


    args = vars(parser.parse_args())

    trainer = BaseNNTrainer(num_pixels=args['num_pixels'],
                              MSEloss=args['mse_loss'],
                              no_epochs=args['no_epochs'],
                              batch_size=args['batch_size'],
                              lr=args['learning_rate'])
    trainer.train_test()


if __name__ == '__main__':
    main()



