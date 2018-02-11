from typing import List, Dict
from urllib.error import HTTPError
from pymlb.data import DeepBallConnector


class GameLogs(DeepBallConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list_games(self, season: int, field_list: List = None):
        return self._get("gamelogs/" + str(season), {"fields": field_list}, cache=True)['listing']

    def list_events(self, season: int, field_list: List = None, search: Dict[str, str] = None):
        parameters = {"fields": field_list}
        if search is not None:
            parameters.update(search)
        return self._get("gamelogs/" + str(season) + "/*/events", parameters, cache=True)['listing']
