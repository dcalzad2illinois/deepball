from pymlb.data import AggregateQueries, VectorSlots, AQStatType, AQGroupType, AQTimeDuration
import numpy as np


def group(data, column_index):
    values = set(data[:, column_index])
    data_without_values = np.concatenate([data[:, :column_index], data[:, column_index + 1:]], axis=-1).astype(int)
    return {value: np.sum(data_without_values[np.argwhere(data[:, column_index] == value)[:, 0]]) for value in values}


slots = VectorSlots(token="[hidden]", cache_directory="data_cache")
aq = AggregateQueries(token="[hidden]", cache_directory="data_cache")

stadium_park_factors = slots.get_vector_slots(6, "ls_out_park_factors")


# get the mapping of batter/season => stadium => PAs in that stadium that year
def get_player_stadiums():
    player_games = aq.new_aggregate_query(AQTimeDuration.GAME, AQGroupType.PLAYER, AQStatType.BATTING,
                                          parameters={"fields": ["batter", "game_id", "pa"], "pa": ">0"})
    player_games = AggregateQueries.query_to_matrices(player_games, lambda row: row["batter"] + row["game_id"][3:7],
                                                      field_list=["pa"],
                                                      additional_fields=[lambda row: row["game_id"][:3]])

    return {
        batter_season: group(player_games[batter_season], column_index=1) for batter_season in player_games.keys()
    }

player_season_stadiums = get_player_stadiums()

# find the park factors for that batter weighted by their PAs in each stadium
key_park_factors = {
    key: np.sum(np.array([value * np.array(stadium_park_factors[stadium][key[8:]]) for stadium, value in stadium_counts.items()]), axis=0) / sum(stadium_counts.values())
    for key, stadium_counts in player_season_stadiums.items()
}

# now add the players that had 0 PA in a season
empty_players = aq.new_aggregate_query(AQTimeDuration.SEASON, AQGroupType.PLAYER, AQStatType.BATTING,
                                          parameters={"fields": ["batter", "season"], "pa": "0"})
for values in empty_players:
    key_park_factors[values["batter"] + str(values["season"])] = np.zeros((7,))

# convert this into a format that the vector slots can handle
player_season_park_factors = {}
for key, factors in key_park_factors.items():
    if key[:8] not in player_season_park_factors:
        player_season_park_factors[key[:8]] = {}
    player_season_park_factors[key[:8]][key[8:]] = factors

# save it!
slots.put_vector_slots(17, "ls_batter_out_park_factors", player_season_park_factors)
