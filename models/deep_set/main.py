
from utilities import MultiSetTrainer
def main():

    trainer = MultiSetTrainer(num_pixels=1500)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()


