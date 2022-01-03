SELECT 
    E.game_code AS "game_code",
    E.game_date AS "game_date",
    E.season AS "season",
    G.home_team_id AS "home_team_id",
    G.vis_team_id AS "away_team_id",
    g.home_score AS "home_final_score",
    g.vis_score AS "away_final_score",
    CASE WHEN g.home_score > g.vis_score THEN 'W'
        WHEN g.home_score < g.vis_score THEN 'L'
        WHEN g.home_score = g.vis_score THEN 'T' END as "home_team_outcome",
    E.nevent AS "nevent",
    E.quarter AS "quarter",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN G.home_score - E.off_start_score ELSE G.home_score - E.def_start_score END AS "home_rest_of_game_score",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN G.vis_score - E.off_start_score ELSE G.vis_score - E.def_start_score END AS "away_rest_of_game_score",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN 1 ELSE 0 END AS "home_team_has_ball",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN E.off_start_score ELSE E.def_start_score END AS "home_start_score",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN E.off_start_score ELSE E.def_start_score END AS "away_start_score",
    E.yd_from_goal AS "yd_from_goal",
    E.from_scrimmage AS "from_scrimmage",
    CASE WHEN E.down IS NULL THEN 0 
        ELSE E.down END AS "down",
    CASE WHEN E.ytg IS NULL THEN -1 
        ELSE E.ytg END AS "ytg",
    E.play_start_time AS "play_start_time"
FROM customer_data.cd_football_events E
    LEFT JOIN customer_data.cd_football_game G
        ON G.game_code = E.game_code
WHERE G.split_number = 0
    AND g.season >= 2008