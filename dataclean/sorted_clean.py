import pandas as pd
DATA_DIR = "./data"

DATA_1995_CSV = DATA_DIR + "/pp-cleaned1995.csv"
DATA_1996_CSV = DATA_DIR + "/pp-cleaned1996.csv"
DATA_1997_CSV = DATA_DIR + "/pp-cleaned1997.csv"
DATA_1998_CSV = DATA_DIR + "/pp-cleaned1998.csv"
DATA_1999_CSV = DATA_DIR + "/pp-cleaned1999.csv"
DATA_2000_CSV = DATA_DIR + "/pp-cleaned2000.csv"
DATA_2001_CSV = DATA_DIR + "/pp-cleaned2001.csv"
DATA_2002_CSV = DATA_DIR + "/pp-cleaned2002.csv"
DATA_2003_CSV = DATA_DIR + "/pp-cleaned2003.csv"
DATA_2004_CSV = DATA_DIR + "/pp-cleaned2004.csv"
DATA_2005_CSV = DATA_DIR + "/pp-cleaned2005.csv"
DATA_2006_CSV = DATA_DIR + "/pp-cleaned2006.csv"
DATA_2007_CSV = DATA_DIR + "/pp-cleaned2007.csv"
DATA_2008_CSV = DATA_DIR + "/pp-cleaned2008.csv"
DATA_2009_CSV = DATA_DIR + "/pp-cleaned2009.csv"
DATA_2010_CSV = DATA_DIR + "/pp-cleaned2010.csv"
DATA_2011_CSV = DATA_DIR + "/pp-cleaned2011.csv"
DATA_2012_CSV = DATA_DIR + "/pp-cleaned2012.csv"
DATA_2013_CSV = DATA_DIR + "/pp-cleaned2013.csv"
DATA_2014_CSV = DATA_DIR + "/pp-cleaned2014.csv"
DATA_2015_CSV = DATA_DIR + "/pp-cleaned2015.csv"
DATA_2016_CSV = DATA_DIR + "/pp-cleaned2016.csv"
DATA_2017_CSV = DATA_DIR + "/pp-cleaned2017.csv"
DATA_2018_CSV = DATA_DIR + "/pp-cleaned2018.csv"
DATA_2019_CSV = DATA_DIR + "/pp-cleaned2019.csv"
DATA_2020_CSV = DATA_DIR + "/pp-cleaned2020.csv"
DATA_2021_CSV = DATA_DIR + "/pp-cleaned2021.csv"



DATA_1995 = pd.read_csv(DATA_1995_CSV)
DATA_1996 = pd.read_csv(DATA_1996_CSV)
DATA_1997 = pd.read_csv(DATA_1997_CSV)
DATA_1998 = pd.read_csv(DATA_1998_CSV)
DATA_1999 = pd.read_csv(DATA_1999_CSV)
DATA_2000 = pd.read_csv(DATA_2000_CSV)
DATA_2001 = pd.read_csv(DATA_2001_CSV)
DATA_2002 = pd.read_csv(DATA_2002_CSV)
DATA_2003 = pd.read_csv(DATA_2003_CSV)
DATA_2004 = pd.read_csv(DATA_2004_CSV)
DATA_2005 = pd.read_csv(DATA_2005_CSV)
DATA_2006 = pd.read_csv(DATA_2006_CSV)
DATA_2007 = pd.read_csv(DATA_2007_CSV)
DATA_2008 = pd.read_csv(DATA_2008_CSV)
DATA_2009 = pd.read_csv(DATA_2009_CSV)
DATA_2010 = pd.read_csv(DATA_2010_CSV)
DATA_2011 = pd.read_csv(DATA_2011_CSV)
DATA_2012 = pd.read_csv(DATA_2012_CSV)
DATA_2013 = pd.read_csv(DATA_2013_CSV)
DATA_2014 = pd.read_csv(DATA_2014_CSV)
DATA_2015 = pd.read_csv(DATA_2015_CSV)
DATA_2016 = pd.read_csv(DATA_2016_CSV)
DATA_2017 = pd.read_csv(DATA_2017_CSV)
DATA_2018 = pd.read_csv(DATA_2018_CSV)
DATA_2019 = pd.read_csv(DATA_2019_CSV)
DATA_2020 = pd.read_csv(DATA_2020_CSV)
DATA_2021 = pd.read_csv(DATA_2021_CSV)



DATA_ALL = pd.concat([DATA_1995, DATA_1996, DATA_1997, DATA_1998, DATA_1999, 
                  DATA_2000, DATA_2001, DATA_2002, DATA_2003, DATA_2004, 
                  DATA_2005, DATA_2006, DATA_2007, DATA_2008, DATA_2009, 
                  DATA_2010, DATA_2011, DATA_2012, DATA_2013, DATA_2014, 
                  DATA_2015, DATA_2016, DATA_2017, DATA_2018, DATA_2019, 
                  DATA_2020, DATA_2021], ignore_index=True)

DATA_ALL.to_csv(DATA_DIR + "/pp-all.csv", index=False)
