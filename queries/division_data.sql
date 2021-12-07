SELECT 
CAST(substr(season_id, 1, 4) AS INT) AS "season",
team_id AS "team_id",
S.league_id AS "league_id",
S.conference_id AS "conference_id",
S.division_id AS "division_id"
FROM football.football_team_season S
WHERE s.league_id = 8
AND CAST(substr(season_id, 1, 4) AS INT) >= 2015