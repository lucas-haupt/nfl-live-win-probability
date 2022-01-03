SELECT
    G.game_code AS "game_code",
    G.casino_id AS "casino_id",
    G.casino_name AS "casino_name",
    G.status AS "status",
    G.season AS "season",
    G.game_type_desc AS "game_type_desc",
    G.game_date AS "game_date",
    G.home_team_name || ' ' || home_team_nickname AS "home_team",
    G.away_team_name || ' ' || away_team_nickname AS "away_team",
    G.cur_favorite_id AS "cur_favorite_id",
    CASE WHEN G.cur_favorite_id = 1 THEN G.cur_spread 
        ELSE -G.cur_spread END AS "cur_spread",
    G.cur_over_under AS "cur_over_under",
    CASE WHEN G.cur_home_money_line <= -100 THEN 1 / (1-100 / G.cur_home_money_line) 
        ELSE 1 / (1 + G.cur_home_money_line / 100) 
        END AS "home_odds",
    CASE WHEN G.cur_away_money_line <= -100 THEN 1 / (1 - 100 / G.cur_away_money_line) 
        ELSE 1 / (1 + G.cur_away_money_line / 100) 
        END AS "away_odds",
    CASE WHEN G.cur_favorite_money_line <= -100 THEN 1 / (1-100 / G.cur_favorite_money_line) 
        ELSE 1 / (1 + G.cur_favorite_money_line / 100) 
        END AS "favorite_odds",
    CASE WHEN G.cur_underdog_money_line <= -100 THEN 1 / (1-100 / G.cur_underdog_money_line) 
        ELSE 1 / (1 + G.cur_underdog_money_line / 100) 
        END AS "underdog_odds"

FROM customer_data.cd_gaming_games G
WHERE G.league_id = 8
    AND G.game_type_id NOT IN (0, 5)
    AND G.season >= 2008
    AND G.cur_home_money_line IS NOT NULL
    AND G.scope_id = 1
    AND G.cur_home_money_line != 0
    AND G.cur_away_money_line != 0
    ORDER BY G.season, G.game_code