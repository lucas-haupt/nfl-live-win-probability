SELECT 
    E.game_code AS "game_code",
    E.game_date AS "game_date",
    G.home_team_id AS "home_team_id",
    G.vis_team_id AS "away_team_id",
    g.home_score AS "home_final_score",
    g.vis_score AS "away_final_score",
    CASE WHEN g.home_score > g.vis_score THEN 'W'
        WHEN g.home_score < g.vis_score THEN 'L'
        WHEN g.home_score = g.vis_score THEN 'T' END as "home_team_outcome",
    E.off_team_id AS "off_team_id",
    E.off_team_name AS "off_team_name",
    E.off_team_abbrev AS "off_team_abbrev",
    E.off_team_nickname AS "off_team_nickname",
    E.def_team_id AS "def_team_id",
    E.def_team_name AS "def_team_name",
    E.def_team_abbrev AS "def_team_abbrev",
    E.def_team_nickname AS "def_team_nickname",
    E.play_unique_id AS "play_unique_id",
    E.nevent AS "nevent",
    E.event_id AS "event_id",
    E.event_name AS "event_name",
    E.eff_cnts AS "eff_cnts",
    E.quarter AS "quarter",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN G.home_score - E.off_start_score ELSE G.home_score - E.def_start_score END AS "home_rest_of_game_score",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN G.vis_score - E.off_start_score ELSE G.vis_score - E.def_start_score END AS "away_rest_of_game_score",

    CASE WHEN E.off_team_id = G.home_team_id
        THEN 1 ELSE 0 END AS "home_team_has_ball",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN E.off_start_score ELSE E.def_start_score END AS "home_start_score",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN E.off_end_score ELSE E.def_end_score END AS "home_end_score",


    CASE WHEN E.off_team_id = G.vis_team_id
        THEN E.off_start_score ELSE E.def_start_score END AS "away_start_score",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN E.off_end_score ELSE E.def_end_score END AS "away_end_score",
    
    E.off_start_score AS "off_start_score",
    E.off_end_score AS "off_end_score",
    E.def_start_score AS "def_start_score",
    E.def_end_score AS "def_end_score",
    E.yd_from_goal AS "yd_from_goal",
    E.is_twomin AS "is_twomin",
    E.from_scrimmage AS "from_scrimmage",
    E.is_first_down AS "is_first_down",
    E.is_scoring_play AS "is_scoring_play",
    E.continuation AS "continuation",
    E.drive_id AS "drive_id",
    E.drive_start AS "drive_start",
    CASE WHEN E.down IS NULL THEN 0 
        ELSE E.down END AS "down",
    CASE WHEN E.ytg IS NULL THEN -1 
        ELSE E.ytg END AS "ytg",
    E.time_of_score AS "time_of_score",
    E.total_play_yd AS "total_play_yd",
    E.timeouts_left AS "timeouts_left",
    E.unit AS "unit",
    E.league_id AS "league_id",
    E.league_name AS "league_name",
    E.off_conference_id AS "off_conference_id",
    E.off_conference_name AS "off_conference_name",
    E.season_id AS "season_id",
    E.season AS "season",
    E.game_type_id AS "game_type_id",
    E.game_type_desc AS "game_type_desc",
    E.def_conference_id AS "def_conference_id",
    E.def_conference_name AS "def_conference_name",
    E.qb_id AS "qb_id",
    E.play_start_time AS "play_start_time",
    E.snap_hash AS "snap_hash"
FROM customer_data.cd_football_events E
    LEFT JOIN customer_data.cd_football_game G
        ON G.game_code = E.game_code
WHERE G.split_number = 0
    AND g.season >= 2008