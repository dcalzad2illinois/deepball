from pymlb.data import AggregateQueries, VectorSlots, AQStatType, AQGroupType, AQTimeDuration
import numpy as np
import numpy.linalg as la


def one_hot_columns(matrix, columns):
    # get the distinct values
    mapping = {}
    distinct_values = {}
    for col_index in columns:
        column = matrix[:, col_index].tolist()
        distinct = list(set(column))
        distinct_values[col_index] = distinct

        new_matrix = np.zeros((matrix.shape[0], len(distinct)))
        column_integers = [distinct.index(entry) for entry in column]
        new_matrix[np.arange(len(column)), column_integers] = 1
        mapping[col_index] = new_matrix

    # turn the untouched columns into column vectors
    for col_index in range(matrix.shape[1]):
        if col_index not in columns:
            mapping[col_index] = np.reshape(matrix[:, col_index], (-1, 1))

    # create the final matrix
    final = np.hstack([item[1] for item in sorted(mapping.items())]).astype(float)
    return final, distinct_values


connector = AggregateQueries(token="[hidden]", cache_directory="data_cache")

team_games = connector.new_aggregate_query(AQTimeDuration.GAME, AQGroupType.TEAM, AQStatType.BATTING)
team_games = AggregateQueries.query_to_matrices(team_games, lambda row: row["game_id"][3:7],
                                                field_list=["batting_team", "pitching_team", "dh_used",
                                                            "pitcher_outs_recorded",
                                                            "k", "ubb", "singles", "doubles", "triples", "hrs",
                                                            "runs"],
                                                additional_fields=[lambda row: row["game_id"][:3]])

all_factors = {}
all_team_batting = {}
all_team_pitching = {}
for season, season_matrix in sorted(team_games.items()):
    season = int(season)

    # reorder the columns to put the home team in the 2nd index column
    season_matrix = np.hstack([season_matrix[:, 0:2], season_matrix[:, -1:], season_matrix[:, 2:-1]])

    # get the Y values
    y = season_matrix[:, 4:].astype(float)

    # divide them all by the pitcher_outs_recorded column
    y = y / y[:, 0:1] * 27

    # get rid of the pitcher_outs_recorded since they're all the same now
    y = y[:, 1:]

    # standardize the output fields
    y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)

    # get the X values
    X = season_matrix[:, 0:4]

    X, encodings = one_hot_columns(X, [0, 1, 2])

    # add a row to the matrix for each feature to regularize it
    regularizer = np.identity(X.shape[1]) * 16
    for i in range(len(encodings[0]) + len(encodings[1]), regularizer.shape[0] - 1):
        regularizer[i, i] = 1

    # if there are no DH's this season, add a regularizer to the DH column so the matrix isn't singular
    if np.mean(X[:, -1]) == 0:
        regularizer[-1, -1] = 1
    else:
        regularizer[-1, -1] = 0

    X = np.vstack([X, regularizer])
    y = np.vstack([y, np.zeros((X.shape[1], y.shape[1]))])

    XTX = X.T.dot(X)
    XTy = X.T.dot(y)
    w = la.solve(XTX, XTy)

    factors = w[len(encodings[0]) + len(encodings[1]):-1, :]
    factors = (factors - np.mean(factors, axis=0, keepdims=True)) / np.std(factors, axis=0, keepdims=True)

    for park, factor in zip(encodings[2], factors.tolist()):
        if park not in all_factors.keys():
            all_factors[park] = {}
        all_factors[park][season] = factor

    team_batting = w[:len(encodings[0]), :]
    team_batting = (team_batting - np.mean(team_batting, axis=0, keepdims=True))

    for team, factor in zip(encodings[0], team_batting.tolist()):
        if team not in all_team_batting.keys():
            all_team_batting[team] = {}
        all_team_batting[team][season] = factor

    team_pitching = w[len(encodings[0]):len(encodings[0]) + len(encodings[1]), :]
    team_pitching = (team_pitching - np.mean(team_pitching, axis=0, keepdims=True))

    for team, factor in zip(encodings[0], team_pitching.tolist()):
        if team not in all_team_pitching.keys():
            all_team_pitching[team] = {}
        all_team_pitching[team][season] = factor

        # # find the correlation between batting team and their park factor (ideally, it should be 0)
        # correlations = []
        # for data, items in [[all_team_batting, encodings[0]], [all_team_pitching, encodings[1]]]:
        #     team_ratings = []
        #     park_factor = []
        #     for team in items:
        #         team_ratings.append(data[team][season])
        #         park_factor.append(all_factors[team][season])
        #     correlations.append(np.corrcoef(np.array(team_ratings), np.array(park_factor))[0, 1])
        # print(str(season) + ": " + str(correlations))


# print("--- Batting ---")
# for team in sorted(all_team_batting.keys()):
#     print(team + ": " + str(all_team_batting[team]))
#
# print("--- Pitching ---")
# for team in sorted(all_team_pitching.keys()):
#     print(team + ": " + str(all_team_pitching[team]))
#
# print("--- Park ---")
# for park in sorted(all_factors.keys()):
#     print(park + ": " + str(all_factors[park]))

def r_squared(factors, ddof=1):
    factors = {k: v for k, v in factors.items() if len(v) >= 2}

    overall_variance = np.var([value for value_list in factors.values() for value in value_list.values()], ddof=ddof)
    sample_variances = sum(
        np.var(list(value_list.values()), ddof=ddof) * len(value_list) for value_list in factors.values()) / sum(
        len(value_list) for value_list in factors.values())
    return 1 - sample_variances / overall_variance


# print("Old R^2 = " + str(r_squared(espn_factors, ddof=1)))
# print("New R^2 = " + str(r_squared(all_factors, ddof=1)))

def keys_to_strings(dictionary):
    return dict(
        (k, dict((str(k2), np.array(v2)) for k2, v2 in v.items()))
        for k, v in dictionary.items()
    )


# create the input park factors
in_factors = {}
for team in all_factors:
    all_factors[team][2018] = np.zeros((7,))  # make it create park factors (in and out) for the current season
    all_factors[team][2019] = np.zeros((7,))  # make it create park factors (in and out) for the current season
    all_factors[team][2020] = np.zeros((7,))  # make it create park factors (in and out) for the current season
    all_factors[team][2021] = np.zeros((7,))  # make it create park factors (in and out) for the current season
    in_factors[team] = {}
    for season, factors in all_factors[team].items():
        sum_factors = np.zeros_like(factors)
        sum_weights = 1
        for i in range(season - 3, season):
            if i in all_factors[team]:
                sum_factors += np.array(all_factors[team][i]) * (4 + i - season)
                sum_weights += 4 + i - season
        in_factors[team][season] = (sum_factors if sum_weights == 0 else sum_factors / sum_weights).tolist()


def shift(factors, delta: int = 1):
    new_factors = {}
    for team in factors:
        new_factors[team] = {}
        first_season = min(factors[team].keys())
        for season, season_factors in sorted(factors[team].items()):
            if season < first_season + delta:
                new_factors[team][season] = np.zeros_like(season_factors)
            new_factors[team][season + delta] = season_factors
    return new_factors


# save them to the database
connector = VectorSlots(connector=connector)
connector.put_vector_slots(6, "ls_in_park_factors", keys_to_strings(in_factors))
connector.put_vector_slots(6, "ls_in_park_factors_shift2", keys_to_strings(shift(in_factors, delta=1)))
connector.put_vector_slots(6, "ls_in_park_factors_shift3", keys_to_strings(shift(in_factors, delta=2)))
connector.put_vector_slots(6, "ls_out_park_factors", keys_to_strings(all_factors))
connector.put_vector_slots(6, "ls_batting_factors", keys_to_strings(all_team_batting))
connector.put_vector_slots(6, "ls_pitching_factors", keys_to_strings(all_team_pitching))
