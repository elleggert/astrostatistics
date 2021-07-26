
from utilities import MultiSetTrainer
def main():

    trainer = MultiSetTrainer(num_pixels=20000)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()


