import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# скрипт предназначен для создания paitplot
# из jupyter работает очень долго


def main() -> None:
    # load the data
    all_data = pd.read_excel('data/Задачи для кандидатов.xlsx', sheet_name='Data')
    all_data = all_data.drop(['index'], axis='columns')
    
    # create and save pairplot
    pair_plot = sns.pairplot(all_data, hue='target', corner=True)
    pair_plot.savefig('images/pair_plot.png')
    return


if __name__ == '__main__':
    main()
