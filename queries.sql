SELECT g.game_id, g.game_date, p.play_id, p.event_type, p.player_id
FROM game_info g
JOIN play_by_play p
  ON g.game_id = p.game_id
WHERE g.game_date > '2010-01-01'
LIMIT 100
