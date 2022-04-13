SELECT 
    E.game_code AS "game_code",
    E.game_date AS "game_date",
    E.season AS "season",
    G.home_team_id AS "home_team_id",
    g.home_team_name || ' ' || g.home_team_nickname AS "home_team",
    g.home_team_abbrev AS "home_team_abbrev",
    G.vis_team_id AS "away_team_id",
    g.vis_team_name || ' ' || g.vis_team_nickname AS "away_team",
    g.vis_team_abbrev AS "away_team_abbrev",
    g.home_score AS "home_final_score",
    g.vis_score AS "away_final_score",
    g.home_score - g.vis_score AS "final_score_diff",
    CASE when G.quarters > 5 THEN 0
        ELSE g.home_score - g.vis_score END AS "end_of_regulation_score_diff",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN G.home_score - E.off_start_score ELSE G.home_score - E.def_start_score END AS "home_rest_of_game_score",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN G.vis_score - E.off_start_score ELSE G.vis_score - E.def_start_score END AS "away_rest_of_game_score",
    CASE when G.quarters > 5 THEN (CASE WHEN E.off_team_id = G.home_team_id THEN 1 ELSE -1 END) * (E.off_start_score - E.def_start_score)
        ELSE (g.home_score - g.vis_score) - (CASE WHEN E.off_team_id = G.home_team_id THEN 1 ELSE -1 END) * (E.off_start_score - E.def_start_score) END AS "end_of_regulation_score_diff_change",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN E.off_end_score - E.off_start_score ELSE E.def_end_score - E.def_start_score END AS "home_score_added",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN E.off_end_score - E.off_start_score ELSE E.def_end_score - E.def_start_score END AS "away_score_added",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN E.off_start_score - E.def_start_score ELSE E.def_start_score - E.off_start_score END AS "current_score_diff",
    e.off_start_score + e.def_start_score AS "current_score_total",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN E.off_start_score ELSE E.def_start_score END AS "home_start_score",
    CASE WHEN E.off_team_id = G.vis_team_id
        THEN E.off_start_score ELSE E.def_start_score END AS "away_start_score",
    CASE WHEN g.home_score > g.vis_score THEN 'W'
        WHEN g.home_score < g.vis_score THEN 'L'
        WHEN g.home_score = g.vis_score THEN 'T' END as "home_team_outcome",
    CASE WHEN g.home_score > g.vis_score THEN 1
        WHEN g.home_score < g.vis_score THEN 0
        WHEN g.home_score = g.vis_score THEN 0 END as "home_team_win",
    CASE WHEN g.home_score > g.vis_score THEN 0
        WHEN g.home_score < g.vis_score THEN 0
        WHEN g.home_score = g.vis_score THEN 1 END as "draw",
    CASE WHEN g.home_score > g.vis_score THEN 0
        WHEN g.home_score < g.vis_score THEN 1
        WHEN g.home_score = g.vis_score THEN 0 END as "away_team_win",
    E.nevent AS "nevent",
    E.quarter AS "quarter",
    CASE WHEN E.quarter >=5 THEN 1
        ELSE 0 END AS "overtime",
    CASE WHEN E.off_team_id = G.home_team_id
        THEN 1 ELSE 0 END AS "home_team_has_ball",
    e.off_team_id AS "off_team_id",
    e.def_team_id AS "def_team_id",
    CASE WHEN E.event_id IN (5, 6, 21, 28, 41, 46, 25) THEN 1
        ELSE 0 END AS "kick_off",
    CASE WHEN E.event_id IN (7, 8, 18, 24) THEN 1
        ELSE 0 END AS "punt",
    CASE WHEN E.event_id IN (22, 47) THEN 1
        ELSE 0 END AS "point_after_kick",        
    CASE WHEN E.event_id IN (49, 52, 53, 54, 55, 56) THEN 1
            ELSE 0 END AS "two_point_attempt",
    CASE WHEN E.event_id IN (17, 35, 48, 50) THEN 1
        ELSE 0 END AS "field_goal_attempt",
    e.off_start_score AS "off_start_score",
    e.off_end_score AS "off_end_score",
    e.off_end_score - e.off_start_score AS "off_score_change",
    e.def_start_score AS "def_start_score",
    e.def_end_score AS "def_end_score",
    e.def_end_score - e.def_start_score AS "def_score_change",
    decode(e.play_cnts, 'T', 1, 'F', 0) AS "play_counts",
    decode(e.eff_cnts, 'T', 1, 'F', 0) AS "efficiency_counts",
    decode(E.from_scrimmage, 'T', 1, 'F', 0) AS "from_scrimmage",
    decode(e.is_first_down, 'T', 1, 'F', 0) AS "first_down",
    decode(e.is_first_down, 'T', 1, 'F', 0) AS "scoring_play",
    decode(E.pos_change, 'T', 1, 'F', 0) AS "possession_change",
    decode(E.continuation, 'T', 1, 'F', 0) AS "continuation",
    E.event_name AS "event_name",
    E.event_id AS "event_id",
    E.yd_gained AS "yards_gained",
    d.end_event AS "drive_outcome_id",
    eec.name AS "drive_outcome_desc",
    CASE WHEN E.down IS NULL THEN 0 
        ELSE E.down END AS "down",
    CASE WHEN E.ytg IS NULL THEN -1 
        ELSE E.ytg END AS "ytg",
    E.yd_from_goal AS "yd_from_goal",
    e.drive_id AS "drive_id",
    e.drive_start AS "drive_start",
    E.play_start_time AS "play_start_time"
FROM customer_data.cd_football_events E
    LEFT JOIN customer_data.cd_football_game G
        ON G.game_code = E.game_code
    LEFT JOIN customer_data.cd_football_drives D
        ON e.game_code = d.game_code 
            AND e.drive_id = d.drive_id
            AND e.off_team_id = d.off_team_id
    LEFT JOIN FOOTBALL.football_event_codes eec
        ON d.end_event = eec.event_code_id
WHERE G.split_number = 0
    AND g.league_id = 8
    AND g.season >= 2008
    AND (d.xp_drive = 'N' OR d.xp_drive is null)
ORDER BY e.game_date, e.game_code, e.nevent
