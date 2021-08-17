import pandas as pd
import tensorflow as tf

# Reading the dataset

csv_list = ["data/2020_play_by_play.csv", "data/2018_play_by_play.csv", "data/2017_play_by_play.csv", "data/2016_play_by_play.csv", "data/2015_play_by_play.csv", "data/2014_play_by_play.csv", "data/2013_play_by_play.csv"]

table = []

for filename in csv_list:
    table.append(pd.read_csv(filename, error_bad_lines=False))

table = pd.concat(table)

#table = pd.read_csv("data/2020_play_by_play.csv")
table2 = pd.read_csv("data/2019_play_by_play.csv")

# Function to define successful play. Returns 1 if successful, 0 if not
def successful_play(df):
    if df['ToGo']<=df['Yards']:
        return 1
    elif df['Down'] < 3 and df['Yards']>=4:
        return 1
    elif df['IsPenalty']==1 and (df['PenaltyTeam']!=df['OffenseTeam']):
        return 1
    else :
        return 0

def convertMinutesToSeconds(df):
    time = (df['Minute']*60) + df['Second']
    return time

def playType(df):
    if df['PlayType']=="RUSH" or df['PlayType']=="SCRAMBLE":
        return 0
    elif df['PlayType']=="PASS" or df['PlayType']=="SACK":
        return 1
    elif df['PlayType']=="FIELD GOAL":
        return 2
    else:
        return 3

# Preconditions: down is between 1 and 4, yardsToGo is less than or equal to 20, yard line is from 0 to 20

def main():

    def specific_situation(down, yardsToGo, yardLine, isPass):
        if (isPass):
            return pass_table.loc[(pass_table['ToGo'] == yardsToGo) & (pass_table['Down'] == down) & (
                        pass_table['YardLine'] == 100 - yardLine)]
        else:
            return rush_table.loc[(rush_table['ToGo'] == yardsToGo) & (rush_table['Down'] == down) & (
                        rush_table['YardLine'] == 100 - yardLine)]
    # A table containing only the redzone plays
    compact_table = table.loc[table['YardLine'] >= 80]
    compact_table.insert(len(compact_table.columns), 'Success', "0")
    compact_table.insert(len(compact_table.columns), 'Time', "0")

    # 0 is rush, 1 is pass, 2 is field goal
    compact_table.insert(len(compact_table.columns), 'PType', "0")

    #compact_table = compact_table.loc[(compact_table['ToGo']<compact_table['Yards']) | (compact_table['IsTouchdown']==1),:].assign(Success=1)

    # Changes the success value
    compact_table['Success'] = compact_table.apply(successful_play, axis=1)
    compact_table['Time'] = compact_table.apply(convertMinutesToSeconds, axis=1)
    compact_table['PType'] = compact_table.apply(playType, axis=1)

    compact_table = compact_table.loc[compact_table['PType'] < 3]

    #success_table = compact_table.loc[(compact_table['Down']<3) & (compact_table['Yards']>=4)]
    #print(success_table)
    #print(compact_table['Time'])

    # A table containing all the pass redzone plays
    pass_table = compact_table.loc[(compact_table['PlayType'] == "PASS") | (compact_table['PlayType'] == "SACK")]
    rush_table = compact_table.loc[compact_table['PlayType'] == "RUSH"]
    #print(pass_table)

    #test_table = specific_situation(down=4, yardsToGo=1, yardLine=1, isPass=False) #pass_table.loc[(pass_table['ToGo'] == 6) & (pass_table['Down'] == 2) & (pass_table['YardLine']==83)]

    #training_df: pd.DataFrame = pass_table

    #features = ['ToGo', 'Down', 'YardLine']
    #print(training_df)

    #training_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(training_df[features].values, tf.int32), tf.cast(training_df['Success'].values, tf.int32)))

    compact_table = compact_table[['Quarter', 'Time', 'ToGo', 'Down', 'YardLine', 'PType', 'Success']]

    return compact_table

def getTest():
    compact_table = table2.loc[table2['YardLine'] >= 80]
    compact_table.insert(len(compact_table.columns), 'Success', "0")
    compact_table.insert(len(compact_table.columns), 'Time', "0")
    compact_table.insert(len(compact_table.columns), 'PType', "0")
    # compact_table = compact_table.loc[(compact_table['ToGo']<compact_table['Yards']) | (compact_table['IsTouchdown']==1),:].assign(Success=1)

    # Changes the success value
    compact_table['Success'] = compact_table.apply(successful_play, axis=1)
    compact_table['Time'] = compact_table.apply(convertMinutesToSeconds, axis=1)
    compact_table['PType'] = compact_table.apply(playType, axis=1)

    compact_table = compact_table.loc[compact_table['PType'] < 3]

    # success_table = compact_table.loc[(compact_table['Down']<3) & (compact_table['Yards']>=4)]
    # print(success_table)
    # print(compact_table)

    # A table containing all the pass redzone plays
    #pass_table = compact_table.loc[(compact_table['PlayType'] == "PASS") | (compact_table['PlayType'] == "SACK")]
    #rush_table = compact_table.loc[compact_table['PlayType'] == "RUSH"]
    #print(pass_table)

    #testing_df: pd.DataFrame = pass_table

    compact_table = compact_table[['Quarter', 'Time', 'ToGo', 'Down', 'YardLine', 'PType', 'Success']]

    return compact_table

print(main())
#main()