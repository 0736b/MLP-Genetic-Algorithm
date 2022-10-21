from ga.ga import GA
from utils.datareader import get_dataset
def main():
    dataset = get_dataset('dataset/wdbc.data')
    ga = GA(5, dataset, 50, [30,1,1])
    ga.run()

if __name__ == '__main__':
    main()