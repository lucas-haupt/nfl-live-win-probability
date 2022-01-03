select 
    GAME_CODE as "game_code",
    GAME_DATE as "game_date",
    HOME_TEAM_ID as "home_team_id",
    AWAY_TEAM_ID as "away_team_id",
    SEASON as "season",
    HOME_TEAM_ABBREV as "home_team_abbrev",
    AWAY_TEAM_ABBREV as "away_team_abbrev",
    WEEK as "week"
from CUSTOMER_DATA.cd_football_schedule
where league_id = 8
    and season >= 2008