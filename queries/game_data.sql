select 
    GAME_CODE as "game_code",
    GAME_DATE as "game_date",
    HOME_TEAM_ID as "home_team_id",
    VIS_TEAM_ID as "vis_team_id",
    SEASON as "season",
    HOME_TEAM_ABBREV as "home_team_abbrev",
    VIS_TEAM_ABBREV as "vis_team_abbrev",
    WEEK as "week",
    HOME_SCORE as "home_score",
    VIS_SCORE as "vis_score"
from customer_data.cd_football_game