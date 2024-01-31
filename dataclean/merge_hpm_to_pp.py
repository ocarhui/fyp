import pandas as pd
DATA_DIR = "./data"
HOUSE_SIZE_DIR = "./data/house_size_by_borough"


DATA_1995 = pd.read_csv(DATA_DIR + "/pp-1995.csv")
DATA_1996 = pd.read_csv(DATA_DIR + "/pp-1996.csv")
DATA_1997 = pd.read_csv(DATA_DIR + "/pp-1997.csv")
DATA_1998 = pd.read_csv(DATA_DIR + "/pp-1998.csv")
DATA_1999 = pd.read_csv(DATA_DIR + "/pp-1999.csv")
DATA_2000 = pd.read_csv(DATA_DIR + "/pp-2000.csv")
DATA_2001 = pd.read_csv(DATA_DIR + "/pp-2001.csv")
DATA_2002 = pd.read_csv(DATA_DIR + "/pp-2002.csv")
DATA_2003 = pd.read_csv(DATA_DIR + "/pp-2003.csv")
DATA_2004 = pd.read_csv(DATA_DIR + "/pp-2004.csv")
DATA_2005 = pd.read_csv(DATA_DIR + "/pp-2005.csv")
DATA_2006 = pd.read_csv(DATA_DIR + "/pp-2006.csv")
DATA_2007 = pd.read_csv(DATA_DIR + "/pp-2007.csv")
DATA_2008 = pd.read_csv(DATA_DIR + "/pp-2008.csv")
DATA_2009 = pd.read_csv(DATA_DIR + "/pp-2009.csv")
DATA_2010 = pd.read_csv(DATA_DIR + "/pp-2010.csv")
DATA_2011 = pd.read_csv(DATA_DIR + "/pp-2011.csv")
DATA_2012 = pd.read_csv(DATA_DIR + "/pp-2012.csv")
DATA_2013 = pd.read_csv(DATA_DIR + "/pp-2013.csv")
DATA_2014 = pd.read_csv(DATA_DIR + "/pp-2014.csv")
DATA_2015 = pd.read_csv(DATA_DIR + "/pp-2015.csv")
DATA_2016 = pd.read_csv(DATA_DIR + "/pp-2016.csv")
DATA_2017 = pd.read_csv(DATA_DIR + "/pp-2017.csv")
DATA_2018 = pd.read_csv(DATA_DIR + "/pp-2018.csv")
DATA_2019 = pd.read_csv(DATA_DIR + "/pp-2019.csv")
DATA_2020 = pd.read_csv(DATA_DIR + "/pp-2020.csv")
DATA_2021 = pd.read_csv(DATA_DIR + "/pp-2021.csv")

Barking_and_Dagenham = pd.read_csv(HOUSE_SIZE_DIR + "/Barking_and_Dagenham_HouseSpec.csv")
Barnet = pd.read_csv(HOUSE_SIZE_DIR + "/Barnet_HouseSpec.csv")
Bexley = pd.read_csv(HOUSE_SIZE_DIR + "/Bexley_HouseSpec.csv")
Brent = pd.read_csv(HOUSE_SIZE_DIR + "/Brent_HouseSpec.csv")
Bromley = pd.read_csv(HOUSE_SIZE_DIR + "/Bromley_HouseSpec.csv")
Camden = pd.read_csv(HOUSE_SIZE_DIR + "/Camden_HouseSpec.csv")
Croydon = pd.read_csv(HOUSE_SIZE_DIR + "/Croydon_HouseSpec.csv")
Ealing = pd.read_csv(HOUSE_SIZE_DIR + "/Ealing_HouseSpec.csv")
Enfield = pd.read_csv(HOUSE_SIZE_DIR + "/Enfield_HouseSpec.csv")
Greenwich = pd.read_csv(HOUSE_SIZE_DIR + "/Greenwich_HouseSpec.csv")
Hackney = pd.read_csv(HOUSE_SIZE_DIR + "/Hackney_HouseSpec.csv")
Hammersmith_and_Fulham = pd.read_csv(HOUSE_SIZE_DIR + "/Hammersmith_and_Fulham_HouseSpec.csv")
Haringey = pd.read_csv(HOUSE_SIZE_DIR + "/Haringey_HouseSpec.csv")
Harrow = pd.read_csv(HOUSE_SIZE_DIR + "/Harrow_HouseSpec.csv")
Havering = pd.read_csv(HOUSE_SIZE_DIR + "/Havering_HouseSpec.csv")
Hillingdon = pd.read_csv(HOUSE_SIZE_DIR + "/Hillingdon_HouseSpec.csv")
Hounslow = pd.read_csv(HOUSE_SIZE_DIR + "/Hounslow_HouseSpec.csv")
Islington = pd.read_csv(HOUSE_SIZE_DIR + "/Islington_HouseSpec.csv")
Kensington_and_Chelsea = pd.read_csv(HOUSE_SIZE_DIR + "/Kensington_and_Chelsea_HouseSpec.csv")
Kingston_upon_Thames = pd.read_csv(HOUSE_SIZE_DIR + "/Kingston_upon_Thames_HouseSpec.csv")
Lambeth = pd.read_csv(HOUSE_SIZE_DIR + "/Lambeth_HouseSpec.csv")
Lewisham = pd.read_csv(HOUSE_SIZE_DIR + "/Lewisham_HouseSpec.csv")
Merton = pd.read_csv(HOUSE_SIZE_DIR + "/Merton_HouseSpec.csv")
Newham = pd.read_csv(HOUSE_SIZE_DIR + "/Newham_HouseSpec.csv")
Redbridge = pd.read_csv(HOUSE_SIZE_DIR + "/Redbridge_HouseSpec.csv")
Richmond_upon_Thames = pd.read_csv(HOUSE_SIZE_DIR + "/Richmond_upon_Thames_HouseSpec.csv")
Southwark = pd.read_csv(HOUSE_SIZE_DIR + "/Southwark_HouseSpec.csv")
Sutton = pd.read_csv(HOUSE_SIZE_DIR + "/Sutton_HouseSpec.csv")
Tower_Hamlets = pd.read_csv(HOUSE_SIZE_DIR + "/Tower_Hamlets_HouseSpec.csv")
Waltham_Forest = pd.read_csv(HOUSE_SIZE_DIR + "/Waltham_Forest_HouseSpec.csv")
Wandsworth = pd.read_csv(HOUSE_SIZE_DIR + "/Wandsworth_HouseSpec.csv")
Westminster = pd.read_csv(HOUSE_SIZE_DIR + "/Westminster_HouseSpec.csv")
City_of_London = pd.read_csv(HOUSE_SIZE_DIR + "/City_of_London_HouseSpec.csv")

boroughList = [Barking_and_Dagenham, Barnet, Bexley, Brent, Bromley, 
               Camden, Croydon, Ealing, Enfield, Greenwich, Hackney, 
               Hammersmith_and_Fulham, Haringey, Harrow, Havering, Hillingdon, 
               Hounslow, Islington, Kensington_and_Chelsea, Kingston_upon_Thames, 
               Lambeth, Lewisham, Merton, Newham, Redbridge, Richmond_upon_Thames, 
               Southwark, Sutton, Tower_Hamlets, Waltham_Forest, Wandsworth, Westminster, 
               City_of_London]

dataList = [DATA_1995, DATA_1996, DATA_1997, DATA_1998, DATA_1999, DATA_2000, DATA_2001,
            DATA_2002, DATA_2003, DATA_2004, DATA_2005, DATA_2006, DATA_2007, DATA_2008,
            DATA_2009, DATA_2010, DATA_2011, DATA_2012, DATA_2013, DATA_2014, DATA_2015,
            DATA_2016, DATA_2017, DATA_2018, DATA_2019, DATA_2020, DATA_2021]

all_boroughs = pd.concat(boroughList, ignore_index=True)

all_years_data = pd.DataFrame()
for year, dataLoop in zip(range(1995, 2022), dataList):
    dataLoop['Year'] = year
    all_years_data = pd.concat([all_years_data, dataLoop], ignore_index=True)

merged_data = pd.merge(all_years_data, all_boroughs, left_on='ID', right_on='transactionid', how='left')

merged_data['PricePer'] = merged_data['priceper']
merged_data['NumOfRms'] = merged_data['numberrooms']
merged_data['HouseSize'] = merged_data['tfarea']
merged_data['EnergyEfficiency'] = merged_data['CURRENT_ENERGY_EFFICIENCY']
merged_data['BuildDate'] = merged_data['CONSTRUCTION_AGE_BAND']

for year in range(1995, 2022):
    print(f"Saved {year} data")
    year_data = merged_data[merged_data['Year'] == year]
    
    
