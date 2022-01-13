import pandas as pd

# replace filepath to unmodified dataset
df = pd.read_csv("C:/Users/Henning Dieckow/Documents/LSE/AI-Project/heart.csv")


# converter function for non-numeric columns:
def convert_strings(string):
    if string == "ASY":
        value = 0
    elif string == "NAP":
        value = 1
    elif string == "ATA":
        value = 2
    elif string == "TA":
        value = 3
    elif string == "Normal":
        value = 0
    elif string == "LVH":
        value = 1
    elif string == "ST":
        value = 2
    elif string == "Flat":
        value = 0
    elif string == "Up":
        value = 1
    elif string == "Down":
        value = 2
    elif string == "N":
        value = 0
    elif string == "Y":
        value = 1
    elif string == "F":
        value = 0
    elif string == "M":
        value = 1
    else:
        value = string
    return value


# apply conversion to dataframe
df["Sex"] = df["Sex"].apply(convert_strings)
df["ChestPainType"] = df["ChestPainType"].apply(convert_strings)
df["RestingECG"] = df["RestingECG"].apply(convert_strings)
df["ST_Slope"] = df["ST_Slope"].apply(convert_strings)
df["ExerciseAngina"] = df["ExerciseAngina"].apply(convert_strings)


# save dataframe to csv (put your filepath here)
df.to_csv(
    "C:/Users/Henning Dieckow/Documents/LSE/AI-Project/heart_mod.csv", index=False
)
