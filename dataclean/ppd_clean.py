import pandas as pd
DATA_DIR = "./data"

DATA_2018_1_CSV = DATA_DIR + "/pp-2018-part1.csv"
DATA_2018_2_CSV = DATA_DIR + "/pp-2018-part2.csv"
DATA_2019_1_CSV = DATA_DIR + "/pp-2019-part1.csv"
DATA_2019_2_CSV = DATA_DIR + "/pp-2019-part2.csv"
DATA_2020_1_CSV = DATA_DIR + "/pp-2020-part1.csv"
DATA_2020_2_CSV = DATA_DIR + "/pp-2020-part2.csv"
DATA_2021_1_CSV = DATA_DIR + "/pp-2021-part1.csv"
DATA_2021_2_CSV = DATA_DIR + "/pp-2021-part2.csv"

DATA_2018_1 = pd.read_csv(DATA_2018_1_CSV)
DATA_2018_2 = pd.read_csv(DATA_2018_2_CSV)
DATA_2019_1 = pd.read_csv(DATA_2019_1_CSV)
DATA_2019_2 = pd.read_csv(DATA_2019_2_CSV)
DATA_2020_1 = pd.read_csv(DATA_2020_1_CSV)
DATA_2020_2 = pd.read_csv(DATA_2020_2_CSV)
DATA_2021_1 = pd.read_csv(DATA_2021_1_CSV)
DATA_2021_2 = pd.read_csv(DATA_2021_2_CSV)

DATA_2018_1 = DATA_2018_1.query("County == 'GREATER LONDON'")
DATA_2018_2 = DATA_2018_2.query("County == 'GREATER LONDON'")
DATA_2019_1 = DATA_2019_1.query("County == 'GREATER LONDON'")
DATA_2019_2 = DATA_2019_2.query("County == 'GREATER LONDON'")
DATA_2020_1 = DATA_2020_1.query("County == 'GREATER LONDON'")
DATA_2020_2 = DATA_2020_2.query("County == 'GREATER LONDON'")
DATA_2021_1 = DATA_2021_1.query("County == 'GREATER LONDON'")
DATA_2021_2 = DATA_2021_2.query("County == 'GREATER LONDON'")

DATA_2018 = pd.concat([DATA_2018_1, DATA_2018_2])
DATA_2019 = pd.concat([DATA_2019_1, DATA_2019_2])
DATA_2020 = pd.concat([DATA_2020_1, DATA_2020_2])
DATA_2021 = pd.concat([DATA_2021_1, DATA_2021_2])

DATA_2018.to_csv(DATA_DIR + "/pp-2018.csv", index=False)
DATA_2019.to_csv(DATA_DIR + "/pp-2019.csv", index=False)
DATA_2020.to_csv(DATA_DIR + "/pp-2020.csv", index=False)
DATA_2021.to_csv(DATA_DIR + "/pp-2021.csv", index=False)

