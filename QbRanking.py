import pandas as pd
import numpy as np

table = pd.read_csv("data/QBRedZoneData")
table2 = pd.read_csv("data/QBRedZoneData2")

# ranking for inside the 10 yard line
def get_ranking_in_ten(name):
    rank = table.index[table.Name == name]
    if rank.size==0:
        return np.int64(33)
    return rank+1

def get_ranking(name):
    rank = table2.index[table2.Player.str.contains(name)]
    if rank.size==0:
        return np.int64(33)
    return rank+1

def team_affect(team):
    rank = table.loc[table.Team == team, 'Name']
    print(table.loc[table.Team == team, 'GP'].iloc[0])
    if table.loc[table.Team == team, 'GP'].iloc[0] < 10:
        return 0
    ranking = get_ranking_in_ten(rank.iloc[0])
    if ranking > 32:
        return -0.1
    elif ranking > 25:
        return -0.05
    elif ranking > 20:
        return -0.025
    elif ranking > 15:
        return 0
    elif ranking > 10:
        return 0.025
    elif ranking > 5:
        return 0.05
    else:
        return 0.1

#print(get_ranking("John Doe"))

table2.sort_values(by=['Yds'], inplace=True, ascending=False, ignore_index=True)
#print(table2)
#print(table2.Yds.to_string(index=False))
#print(get_ranking("Colt McCoy"))
print(team_affect("CIN"))