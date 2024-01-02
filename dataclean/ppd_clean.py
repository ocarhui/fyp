import pandas as pd
DATA_DIR = "./data"

DATA_1995_1_CSV = DATA_DIR + "/pp-1995-part1.csv"
DATA_1995_2_CSV = DATA_DIR + "/pp-1995-part2.csv"
DATA_1996_1_CSV = DATA_DIR + "/pp-1996-part1.csv"
DATA_1996_2_CSV = DATA_DIR + "/pp-1996-part2.csv"
DATA_1997_1_CSV = DATA_DIR + "/pp-1997-part1.csv"
DATA_1997_2_CSV = DATA_DIR + "/pp-1997-part2.csv"
DATA_1998_1_CSV = DATA_DIR + "/pp-1998-part1.csv"
DATA_1998_2_CSV = DATA_DIR + "/pp-1998-part2.csv"
DATA_1999_1_CSV = DATA_DIR + "/pp-1999-part1.csv"
DATA_1999_2_CSV = DATA_DIR + "/pp-1999-part2.csv"
DATA_2000_1_CSV = DATA_DIR + "/pp-2000-part1.csv"
DATA_2000_2_CSV = DATA_DIR + "/pp-2000-part2.csv"
DATA_2001_1_CSV = DATA_DIR + "/pp-2001-part1.csv"
DATA_2001_2_CSV = DATA_DIR + "/pp-2001-part2.csv"
DATA_2002_1_CSV = DATA_DIR + "/pp-2002-part1.csv"
DATA_2002_2_CSV = DATA_DIR + "/pp-2002-part2.csv"
DATA_2003_1_CSV = DATA_DIR + "/pp-2003-part1.csv"
DATA_2003_2_CSV = DATA_DIR + "/pp-2003-part2.csv"
DATA_2004_1_CSV = DATA_DIR + "/pp-2004-part1.csv"
DATA_2004_2_CSV = DATA_DIR + "/pp-2004-part2.csv"
DATA_2005_1_CSV = DATA_DIR + "/pp-2005-part1.csv"
DATA_2005_2_CSV = DATA_DIR + "/pp-2005-part2.csv"
DATA_2006_1_CSV = DATA_DIR + "/pp-2006-part1.csv"
DATA_2006_2_CSV = DATA_DIR + "/pp-2006-part2.csv"
DATA_2007_1_CSV = DATA_DIR + "/pp-2007-part1.csv"
DATA_2007_2_CSV = DATA_DIR + "/pp-2007-part2.csv"



DATA_1995_1 = pd.read_csv(DATA_1995_1_CSV)
DATA_1995_2 = pd.read_csv(DATA_1995_2_CSV)
DATA_1996_1 = pd.read_csv(DATA_1996_1_CSV)
DATA_1996_2 = pd.read_csv(DATA_1996_2_CSV)
DATA_1997_1 = pd.read_csv(DATA_1997_1_CSV)
DATA_1997_2 = pd.read_csv(DATA_1997_2_CSV)
DATA_1998_1 = pd.read_csv(DATA_1998_1_CSV)
DATA_1998_2 = pd.read_csv(DATA_1998_2_CSV)
DATA_1999_1 = pd.read_csv(DATA_1999_1_CSV)
DATA_1999_2 = pd.read_csv(DATA_1999_2_CSV)
DATA_2000_1 = pd.read_csv(DATA_2000_1_CSV)
DATA_2000_2 = pd.read_csv(DATA_2000_2_CSV)
DATA_2001_1 = pd.read_csv(DATA_2001_1_CSV)
DATA_2001_2 = pd.read_csv(DATA_2001_2_CSV)
DATA_2002_1 = pd.read_csv(DATA_2002_1_CSV)
DATA_2002_2 = pd.read_csv(DATA_2002_2_CSV)
DATA_2003_1 = pd.read_csv(DATA_2003_1_CSV)
DATA_2003_2 = pd.read_csv(DATA_2003_2_CSV)
DATA_2004_1 = pd.read_csv(DATA_2004_1_CSV)
DATA_2004_2 = pd.read_csv(DATA_2004_2_CSV)
DATA_2005_1 = pd.read_csv(DATA_2005_1_CSV)
DATA_2005_2 = pd.read_csv(DATA_2005_2_CSV)
DATA_2006_1 = pd.read_csv(DATA_2006_1_CSV)
DATA_2006_2 = pd.read_csv(DATA_2006_2_CSV)
DATA_2007_1 = pd.read_csv(DATA_2007_1_CSV)
DATA_2007_2 = pd.read_csv(DATA_2007_2_CSV)

headers = ["ID","Price","Date","Postcode","Type","Newbuild","Ownership","Street Number","Flat Number","Street Name","Area", "Town", "City", "County", "Unnamed: 14", "Unnamed: 15"]
DATA_1995_1.columns = headers
DATA_1995_2.columns = headers
DATA_1996_1.columns = headers
DATA_1996_2.columns = headers
DATA_1997_1.columns = headers
DATA_1997_2.columns = headers
DATA_1998_1.columns = headers
DATA_1998_2.columns = headers
DATA_1999_1.columns = headers
DATA_1999_2.columns = headers
DATA_2000_1.columns = headers
DATA_2000_2.columns = headers
DATA_2001_1.columns = headers
DATA_2001_2.columns = headers
DATA_2002_1.columns = headers
DATA_2002_2.columns = headers
DATA_2003_1.columns = headers
DATA_2003_2.columns = headers
DATA_2004_1.columns = headers
DATA_2004_2.columns = headers
DATA_2005_1.columns = headers
DATA_2005_2.columns = headers
DATA_2006_1.columns = headers
DATA_2006_2.columns = headers
DATA_2007_1.columns = headers
DATA_2007_2.columns = headers

DATA_1995_1 = DATA_1995_1.query("County == 'GREATER LONDON'")
DATA_1995_2 = DATA_1995_2.query("County == 'GREATER LONDON'")
DATA_1996_1 = DATA_1996_1.query("County == 'GREATER LONDON'")
DATA_1996_2 = DATA_1996_2.query("County == 'GREATER LONDON'")
DATA_1997_1 = DATA_1997_1.query("County == 'GREATER LONDON'")
DATA_1997_2 = DATA_1997_2.query("County == 'GREATER LONDON'")
DATA_1998_1 = DATA_1998_1.query("County == 'GREATER LONDON'")
DATA_1998_2 = DATA_1998_2.query("County == 'GREATER LONDON'")
DATA_1999_1 = DATA_1999_1.query("County == 'GREATER LONDON'")
DATA_1999_2 = DATA_1999_2.query("County == 'GREATER LONDON'")
DATA_2000_1 = DATA_2000_1.query("County == 'GREATER LONDON'")
DATA_2000_2 = DATA_2000_2.query("County == 'GREATER LONDON'")
DATA_2001_1 = DATA_2001_1.query("County == 'GREATER LONDON'")
DATA_2001_2 = DATA_2001_2.query("County == 'GREATER LONDON'")
DATA_2002_1 = DATA_2002_1.query("County == 'GREATER LONDON'")
DATA_2002_2 = DATA_2002_2.query("County == 'GREATER LONDON'")
DATA_2003_1 = DATA_2003_1.query("County == 'GREATER LONDON'")
DATA_2003_2 = DATA_2003_2.query("County == 'GREATER LONDON'")
DATA_2004_1 = DATA_2004_1.query("County == 'GREATER LONDON'")
DATA_2004_2 = DATA_2004_2.query("County == 'GREATER LONDON'")
DATA_2005_1 = DATA_2005_1.query("County == 'GREATER LONDON'")
DATA_2005_2 = DATA_2005_2.query("County == 'GREATER LONDON'")
DATA_2006_1 = DATA_2006_1.query("County == 'GREATER LONDON'")
DATA_2006_2 = DATA_2006_2.query("County == 'GREATER LONDON'")
DATA_2007_1 = DATA_2007_1.query("County == 'GREATER LONDON'")
DATA_2007_2 = DATA_2007_2.query("County == 'GREATER LONDON'")

DATA_1995 = pd.concat([DATA_1995_1, DATA_1995_2])
DATA_1996 = pd.concat([DATA_1996_1, DATA_1996_2])
DATA_1997 = pd.concat([DATA_1997_1, DATA_1997_2])
DATA_1998 = pd.concat([DATA_1998_1, DATA_1998_2])
DATA_1999 = pd.concat([DATA_1999_1, DATA_1999_2])
DATA_2000 = pd.concat([DATA_2000_1, DATA_2000_2])
DATA_2001 = pd.concat([DATA_2001_1, DATA_2001_2])
DATA_2002 = pd.concat([DATA_2002_1, DATA_2002_2])
DATA_2003 = pd.concat([DATA_2003_1, DATA_2003_2])
DATA_2004 = pd.concat([DATA_2004_1, DATA_2004_2])
DATA_2005 = pd.concat([DATA_2005_1, DATA_2005_2])
DATA_2006 = pd.concat([DATA_2006_1, DATA_2006_2])
DATA_2007 = pd.concat([DATA_2007_1, DATA_2007_2])

DATA_1995.to_csv(DATA_DIR + "/pp-1995.csv", index=False)
DATA_1996.to_csv(DATA_DIR + "/pp-1996.csv", index=False)
DATA_1997.to_csv(DATA_DIR + "/pp-1997.csv", index=False)
DATA_1998.to_csv(DATA_DIR + "/pp-1998.csv", index=False)
DATA_1999.to_csv(DATA_DIR + "/pp-1999.csv", index=False)
DATA_2000.to_csv(DATA_DIR + "/pp-2000.csv", index=False)
DATA_2001.to_csv(DATA_DIR + "/pp-2001.csv", index=False)
DATA_2002.to_csv(DATA_DIR + "/pp-2002.csv", index=False)
DATA_2003.to_csv(DATA_DIR + "/pp-2003.csv", index=False)
DATA_2004.to_csv(DATA_DIR + "/pp-2004.csv", index=False)
DATA_2005.to_csv(DATA_DIR + "/pp-2005.csv", index=False)
DATA_2006.to_csv(DATA_DIR + "/pp-2006.csv", index=False)
DATA_2007.to_csv(DATA_DIR + "/pp-2007.csv", index=False)

