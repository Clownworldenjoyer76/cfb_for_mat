# Situational Features — Acceptance Checklist

Generated: 2025-10-02T10:15:00.620641Z

## Columns to appear in `data/modeling_dataset.csv`

- **is_home** (int (0/1)): Game is at home venue for the team
- **is_away** (int (0/1)): Game is at opponent's venue
- **is_neutral** (int (0/1)): Game is at neutral site
- **rest_days** (int): Days since team's previous game
- **had_bye** (int (0/1)): Previous interval suggests a bye week (e.g., ≥13 days)
- **short_week** (int (0/1)): Rest_days ≤ 6
- **back_to_back_road** (int (0/1)): Team played away in previous game too
- **third_straight_road** (int (0/1)): Team on third consecutive away game
- **travel_km** (float): Great-circle distance from previous game site to current site
- **long_trip_flag** (int (0/1)): Travel_km exceeds threshold (e.g., 1500km)
- **tz_diff_from_home** (int): Time zone offset (hours) between team's home TZ and venue TZ
- **east_to_west_flag** (int (0/1)): Travel direction from eastern to western time zone
- **west_to_east_flag** (int (0/1)): Travel direction from western to eastern time zone
- **body_clock_kickoff_hour** (int): Kickoff local time converted to team's home time zone
- **altitude_game** (int (0/1)): Venue altitude exceeds threshold (e.g., 1000m)

## Build requirements

- Each row must satisfy: `is_home + is_away + is_neutral == 1`.
- `rest_days` is non-negative; `had_bye` = 1 if rest_days ≥ configured threshold (default 13).
- `travel_km` ≥ 0; median near 0 for home games; distribution right-skewed for teams coming off road trips.
- If venue has a time zone, derive `tz_diff_from_home` and `body_clock_kickoff_hour`; else set to null and log counts.
- For away/neutral games, `is_home` must be 0. For neutral games, both `is_home` and `is_away` must be 0 and `is_neutral` = 1.

## Logging requirements (append to `logs_build_modeling_dataset.txt`)

- Total rows processed; count of home/away/neutral.
- Mean/median of `rest_days` and `travel_km`; count of `had_bye=1`, `short_week=1`.
- Null/unknown counts for venue coordinates and time zones.

## CI acceptance

- Dataset build job exits non-zero if any of the following occur:

  - Any row violates `is_home + is_away + is_neutral == 1`.
  - >5% rows missing venue coords when `is_away` or `is_neutral`.
  - Feature columns missing from output.