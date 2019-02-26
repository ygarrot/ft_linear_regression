from ft_linear_regression import LinearRegression
import argparse

def main():
    parser = argparse.ArgumentParser(description='predict price from mileage')
    parser.add_argument("mileage", type=float, default=0, help="mileage")

    opt = parser.parse_args()
    lr = LinearRegression()
    lr.predict(opt.mileage, "predict.csv")

if __name__ == '__main__':
    main()
