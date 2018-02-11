from enum import Enum, unique


@unique
class AQTimeDuration(Enum):
    GAME = "games"
    SEASON = "seasons"
    CAREER = "careers"


@unique
class AQGroupType(Enum):
    TEAM = "team"
    PLAYER = "player"


@unique
class AQStatType(Enum):
    BATTING = "batting"
    PITCHING = "pitching"
