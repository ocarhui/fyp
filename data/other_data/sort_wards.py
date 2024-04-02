import pandas as pd

data = pd.read_csv('./data/other_data/wards.csv', encoding='ISO-8859-1')  # Assuming ISO-8859-1 encoding; adjust if necessary

# List of corrected postal code prefixes
postcode_prefixes = [
    "E6", "E1", "IG8", "N15", "SW9", "N12", "KT6", "W7", "KT1", "N9", "DA17", "DA6", "RM7",
    "BR4", "SE18", "EC4", "DA1", "BR3", "SE8", "EC3", "E15", "E20", "TW9", "CR5", "SM6", "RM12",
    "CR2", "SM1", "UB8", "DA16", "N8", "W6", "W1", "SE9", "EC2", "BR2", "RM1", "SE19", "DA7",
    "RM6", "E7", "N13", "SW8", "N14", "IG9", "WC1", "UB9", "N22", "RM14", "RM13", "CR4", "TW8",
    "SE28", "E18", "TW5", "E12", "SE22", "TW2", "SE25", "SW10", "NW11", "SW17", "UB4", "EN4",
    "EN3", "EN8", "UB3", "W11", "IG3", "SW2", "N19", "SW5", "N5", "N2", "NW1", "HA6", "TW11",
    "SE13", "SE4", "NW6", "HA1", "SE3", "SE14", "SW16", "NW10", "SW11", "UB2", "EN5", "CR8",
    "UB5", "EN2", "SE24", "E14", "TW3", "SE23", "E13", "TW4", "N3", "N4", "SE15", "SE2", "NW7",
    "HA0", "SE5", "SE12", "TW10", "HA7", "W10", "SW20", "SW4", "N18", "IG5", "SW3", "IG2", "UB7",
    "SW13", "SW14", "E11", "SE21", "E16", "TW1", "SE26", "NW2", "TW12", "HA5", "SE10", "SE7",
    "NW5", "RM8", "HA2", "SE17", "W8", "DA18", "N6", "N1", "KT9", "SW1", "IG7", "SW6", "W12",
    "E9", "SE27", "E17", "SE20", "IG1", "IG4", "TW7", "E10", "UB1", "UB6", "EN1", "SW15", "SW12",
    "SW7", "IG6", "W14", "E8", "W13", "SE16", "SE1", "RM9", "NW4", "HA3", "DA8", "SE6", "SE11",
    "NW3", "HA4", "TW13", "KT8", "N7", "W9", "DA5", "NW9", "BR7", "HA9", "RM3", "TN16", "W3",
    "KT5", "W4", "KT2", "BR5", "BR8", "DA14", "N16", "N11", "E5", "E2", "SM5", "RM11", "SM2",
    "N20", "SW18", "IG11", "UB10", "N10", "N17", "E3", "E4", "EC1", "HA8", "RM2", "BR1", "RM5",
    "BR6", "NW8", "KT3", "DA15", "W5", "WD3", "KT4", "W2", "N21", "CR0", "SM3", "RM10", "CR7",
    "SM4", "WC2", "SW19"
]

# Filter the data
filtered_data = data[data['pcds'].apply(lambda x: any(x.strip().startswith(prefix) for prefix in postcode_prefixes))]
filtered_data = filtered_data.drop(["pcd7", "pcd8", "par11cd", "par11nm", "par11nmw", "wd11nm", "wd11nmw", "lad11nm", "lad11nmw"], axis=1)

# Save the filtered data to a new CSV file
filtered_data.to_csv('./data/other_data/filtered_postcodes.csv', index=False)