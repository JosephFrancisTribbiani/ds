from titanic.db import load_data


def main():
    df = load_data(table="features")
    print(df.shape)
    return


if __name__ == "__main__":
    main()
