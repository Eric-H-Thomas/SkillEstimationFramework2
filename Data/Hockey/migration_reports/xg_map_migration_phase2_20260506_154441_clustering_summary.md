# Phase 2 clustering summary (xG map migration)

This note summarizes the *Phase 2 clustering outputs* (top-N slices per season) used to spot where coverage mismatches concentrate.

Data sources:
- `xg_map_migration_phase2_20260506_154441_cluster_games_top25.csv`
- `xg_map_migration_phase2_20260506_154441_cluster_players_top25.csv`

## Important caveat
These CSVs contain **top-N slices per season** (here N=25) for each clustering view, not the full population.
They are best used to identify *which games/players are most skewed*, but the counts here **are not season totals**.

## Quick stats (within the top-N outputs)
- Games: max `new_only_count`=115, max `legacy_only_count`=115
- Players: max `new_only_count`=138, max `legacy_only_count`=77

## Per-season highlights
### 20212022
- Games (IDs):
  - Highest new-only: game_id=42212 new_only=71 legacy_only=0 overlap=8 shots=96
  - Highest legacy-only: game_id=43397 legacy_only=114 new_only=0 overlap=0 shots=146
- Skew (within top-N games rows): legacy_only > new_only 50%, new_only > legacy_only 50%, ties 0%.
- Players (IDs):
  - Highest new-only: player_id=949964 new_only=34 legacy_only=44 overlap=287 shots=553
  - Highest legacy-only: player_id=950185 legacy_only=77 new_only=21 overlap=319 shots=661
- Skew (within top-N player rows): legacy_only > new_only 86%, new_only > legacy_only 9%, ties 5%.

### 20222023
- Games (IDs):
  - Highest new-only: game_id=268868 new_only=17 legacy_only=18 overlap=171 shots=273
  - Highest legacy-only: game_id=44935 legacy_only=88 new_only=0 overlap=0 shots=136
- Skew (within top-N games rows): legacy_only > new_only 51%, new_only > legacy_only 49%, ties 0%.
- Players (IDs):
  - Highest new-only: player_id=949886 new_only=54 legacy_only=26 overlap=617 shots=757
  - Highest legacy-only: player_id=950163 legacy_only=40 new_only=28 overlap=457 shots=601
- Skew (within top-N player rows): legacy_only > new_only 33%, new_only > legacy_only 67%, ties 0%.

### 20232024
- Games (IDs):
  - Highest new-only: game_id=271464 new_only=115 legacy_only=2 overlap=4 shots=142
  - Highest legacy-only: game_id=2705442 legacy_only=115 new_only=0 overlap=0 shots=152
- Skew (within top-N games rows): legacy_only > new_only 48%, new_only > legacy_only 52%, ties 0%.
- Players (IDs):
  - Highest new-only: player_id=950160 new_only=138 legacy_only=14 overlap=534 shots=788
  - Highest legacy-only: player_id=949167 legacy_only=34 new_only=55 overlap=411 shots=654
- Skew (within top-N player rows): legacy_only > new_only 0%, new_only > legacy_only 100%, ties 0%.

### 20242025
- Games (IDs):
  - Highest new-only: game_id=2775929 new_only=89 legacy_only=1 overlap=13 shots=136
  - Highest legacy-only: game_id=2774809 legacy_only=93 new_only=0 overlap=0 shots=124
- Skew (within top-N games rows): legacy_only > new_only 48%, new_only > legacy_only 52%, ties 0%.
- Players (IDs):
  - Highest new-only: player_id=949745 new_only=129 legacy_only=10 overlap=250 shots=461
  - Highest legacy-only: player_id=949964 legacy_only=44 new_only=41 overlap=591 shots=717
- Skew (within top-N player rows): legacy_only > new_only 13%, new_only > legacy_only 85%, ties 2%.

## Interpretation / next checks
- Games often show extreme single-sided skew in the top-N excerpt (e.g., very large `legacy_only_count` with `overlap_count=0` for some game IDs).
- Players often have both `new_only_count` and `legacy_only_count` non-zero alongside substantial `overlap_count`, suggesting mismatches are not isolated to one player.
- These outputs are grouped by numeric IDs (`GAME_ID_HAWKS`, `PLAYER_ID_HAWKS`). If you want date/name attribution, we can extend the SQL once the exact columns for game date and player name are confirmed in your Snowflake schemas.
