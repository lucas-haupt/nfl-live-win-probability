SELECT 
    game_code AS "game_code",
    game_date AS "game_date",
    home_team_id AS "home_team_id",
    away_team_id AS "away_team_id",
    season AS "season",
    home_team_abbrev AS "home_team_abbrev",
    away_team_abbrev AS "away_team_abbrev",
    week AS "week",
    home_score AS "home_score",
    away_score AS "away_score",
    status AS "status",
    CASE status WHEN 'Final' THEN 11
        WHEN 'Pre-Game' THEN 1 END AS "game_state_id"
FROM customer_data.cd_football_schedule
WHERE league_id = 8
    AND season >=2008
    AND status IN ('Final', 'Pre-Game')
    AND home_team_id != -99
    AND game_type_id NOT in (9, 4)
