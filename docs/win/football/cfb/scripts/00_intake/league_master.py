"""
league_master.py

Builds a league-level master file mapping each CFB team to its
conference and optional division, plus a long-format standings file.

CFB differs from the NFL hierarchy: ESPN groups can contain broad
classification groups, conferences, and (when applicable) child groups.
This script uses ESPN's group hierarchy and isConference flag instead of
assuming conference -> division for every team.

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{season}/types/{type}/groups

Output:
    docs/win/football/cfb/data/master/league_master.csv
    docs/win/football/cfb/data/master/league_standings.csv
"""

import csv
import json
import os
import re
import urllib.request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SEASON = 2026
SEASON_TYPE = 2  # Regular Season

GROUPS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/"
    f"college-football/seasons/{SEASON}/types/{SEASON_TYPE}/groups?limit=1000"
)

TEAM_MASTER_PATH = "docs/win/football/cfb/data/master/team_master.csv"
LEAGUE_MASTER_PATH = "docs/win/football/cfb/data/master/league_master.csv"
LEAGUE_STANDINGS_PATH = "docs/win/football/cfb/data/master/league_standings.csv"

TEAM_ID_PATTERN = re.compile(r"/teams/(\d+)(?:[/?]|$)")
GROUP_ID_PATTERN = re.compile(r"/groups/([^/?]+)(?:[/?]|$)")

MASTER_COLUMNS = [
    "team_id",
    "team_abbr",
    "conference",
    "conference_abbr",
    "division",
    "division_abbr",
    "season",
]

STANDINGS_COLUMNS = [
    "team_id",
    "team_abbr",
    "conference",
    "conference_abbr",
    "division",
    "division_abbr",
    "standings_type",
    "stat_name",
    "stat_value",
    "season",
]


_GROUP_CACHE = {}
_COLLECTION_CACHE = {}


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def fetch_cached(url, timeout=10):
    if url not in _COLLECTION_CACHE:
        _COLLECTION_CACHE[url] = fetch_json(url, timeout=timeout)
    return _COLLECTION_CACHE[url]


def with_limit(url, limit=1000):
    """Increase ESPN collection limits without dropping existing query params."""
    parts = urlsplit(str(url or ""))
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["limit"] = str(limit)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def is_conference_group(group):
    value = group.get("isConference")
    return value is True or str(value).strip().lower() == "true"


def extract_team_id(ref_url):
    match = TEAM_ID_PATTERN.search(str(ref_url or ""))
    return match.group(1) if match else ""


def extract_group_id(ref_url):
    match = GROUP_ID_PATTERN.search(str(ref_url or ""))
    return match.group(1) if match else ""


def group_identity(group, ref_url=""):
    return str(group.get("id", "")).strip() or extract_group_id(ref_url) or str(ref_url or "")


def get_team_id_to_abbr():
    lookup = {}

    with open(TEAM_MASTER_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing = [column for column in ("team_id", "team_abbr") if column not in fieldnames]
        if missing:
            raise RuntimeError(f"team_master.csv missing required columns: {missing}")

        for row in reader:
            team_id = str(row.get("team_id", "")).strip()
            team_abbr = str(row.get("team_abbr", "")).strip()

            if team_id and team_id not in lookup:
                lookup[team_id] = team_abbr

    return lookup


def resolve_group(ref_url):
    ref_url = str(ref_url or "").strip()
    if not ref_url:
        return None

    group_id = extract_group_id(ref_url)
    cache_key = group_id or ref_url

    if cache_key not in _GROUP_CACHE:
        _GROUP_CACHE[cache_key] = fetch_json(ref_url)

    return _GROUP_CACHE[cache_key]


def collection_items(ref_obj):
    if not isinstance(ref_obj, dict):
        return []

    ref_url = str(ref_obj.get("$ref", "")).strip()
    if not ref_url:
        return []

    payload = fetch_cached(with_limit(ref_url))
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def get_child_groups(group):
    children = []

    for item in collection_items(group.get("children", {})):
        child_ref = str(item.get("$ref", "")).strip() if isinstance(item, dict) else ""
        if not child_ref:
            continue

        try:
            child = resolve_group(child_ref)
        except Exception as e:
            print(f"failed to resolve child group {child_ref}: {e}")
            continue

        if child:
            children.append((child_ref, child))

    return children


def get_group_team_ids(group):
    team_ids = []

    for item in collection_items(group.get("teams", {})):
        team_ref = str(item.get("$ref", "")).strip() if isinstance(item, dict) else ""
        team_id = extract_team_id(team_ref)
        if team_id:
            team_ids.append(team_id)

    return team_ids


def parent_ref(group):
    parent = group.get("parent", {})
    if not isinstance(parent, dict):
        return ""
    return str(parent.get("$ref", "")).strip()


def nearest_conference(group, ref_url=""):
    """Return the nearest group marked by ESPN as a conference."""
    current = group
    current_ref = ref_url
    seen = set()

    while isinstance(current, dict):
        identity = group_identity(current, current_ref)
        if identity in seen:
            break
        seen.add(identity)

        if is_conference_group(current):
            return current_ref, current

        next_ref = parent_ref(current)
        if not next_ref:
            break

        try:
            current = resolve_group(next_ref)
        except Exception as e:
            print(f"failed to resolve parent group {next_ref}: {e}")
            break

        current_ref = next_ref

    return "", None


def hierarchy_labels(group, ref_url=""):
    """
    Resolve conference/division labels for a CFB group.

    If the current group is a conference, division stays blank. If the
    current group is below a conference, the current group is retained in
    the legacy division columns. Broad non-conference containers are not
    treated as conferences.
    """
    conf_ref, conference = nearest_conference(group, ref_url)

    if conference:
        conf_name = str(conference.get("name", "")).strip()
        conf_abbr = str(conference.get("abbreviation", "")).strip()

        current_id = group_identity(group, ref_url)
        conference_id = group_identity(conference, conf_ref)

        if current_id == conference_id:
            return conf_name, conf_abbr, "", "", True

        return (
            conf_name,
            conf_abbr,
            str(group.get("name", "")).strip(),
            str(group.get("abbreviation", "")).strip(),
            True,
        )

    return "", "", "", "", False


def standings_payloads(group):
    standings = group.get("standings", {})
    if not isinstance(standings, dict):
        return []

    standings_ref = str(standings.get("$ref", "")).strip()
    if not standings_ref:
        return []

    try:
        root = fetch_cached(standings_ref)
    except Exception as e:
        print(f"failed to resolve standings list for {group.get('name', '')}: {e}")
        return []

    payloads = []

    # Some ESPN standings refs resolve directly to one standings object.
    if isinstance(root.get("standings"), list):
        payloads.append(root)

    # Others resolve to a collection of standings types.
    for item in root.get("items", []) if isinstance(root.get("items", []), list) else []:
        if not isinstance(item, dict):
            continue

        if isinstance(item.get("standings"), list):
            payloads.append(item)
            continue

        type_ref = str(item.get("$ref", "")).strip()
        if not type_ref:
            continue

        try:
            payload = fetch_cached(type_ref)
        except Exception as e:
            print(f"failed to resolve standings type {type_ref}: {e}")
            continue

        if isinstance(payload, dict):
            payloads.append(payload)

    return payloads


def get_standings_rows(group, ref_url, team_id_to_abbr):
    """Return long-format standings rows for one conference/group."""
    conf_name, conf_abbr, div_name, div_abbr, has_conference = hierarchy_labels(group, ref_url)
    if not has_conference:
        return []

    rows = []

    for standings_type in standings_payloads(group):
        type_name = str(
            standings_type.get("name")
            or standings_type.get("displayName")
            or standings_type.get("type")
            or ""
        ).strip()

        team_count = 0

        for team_standing in standings_type.get("standings", []):
            if not isinstance(team_standing, dict):
                continue

            team_ref = str(team_standing.get("team", {}).get("$ref", "")).strip()
            team_id = extract_team_id(team_ref)

            # Keep league output scoped to the teams already accepted into
            # this pipeline's team_master.csv.
            if not team_id or team_id not in team_id_to_abbr:
                continue

            team_count += 1
            team_abbr = team_id_to_abbr.get(team_id, "")

            for record in team_standing.get("records", []):
                if not isinstance(record, dict):
                    continue

                for stat in record.get("stats", []):
                    if not isinstance(stat, dict):
                        continue

                    rows.append({
                        "team_id": team_id,
                        "team_abbr": team_abbr,
                        "conference": conf_name,
                        "conference_abbr": conf_abbr,
                        "division": div_name,
                        "division_abbr": div_abbr,
                        "standings_type": type_name,
                        "stat_name": stat.get("name", ""),
                        "stat_value": stat.get("value", ""),
                        "season": SEASON,
                    })

        print(
            f"  conference={conf_name} "
            f"division={div_name or '-'} "
            f"standings_type={type_name} teams={team_count}"
        )

    return rows


def discover_groups(top_groups):
    """Recursively resolve the group graph and return unique groups."""
    discovered = {}

    def visit(ref_url):
        try:
            group = resolve_group(ref_url)
        except Exception as e:
            print(f"failed to resolve group {ref_url}: {e}")
            return

        if not group:
            return

        identity = group_identity(group, ref_url)
        if identity in discovered:
            return

        discovered[identity] = (ref_url, group)

        upstream_ref = parent_ref(group)
        if upstream_ref:
            visit(upstream_ref)

        for child_ref, _child in get_child_groups(group):
            visit(child_ref)

    for item in top_groups.get("items", []):
        ref_url = str(item.get("$ref", "")).strip() if isinstance(item, dict) else ""
        if ref_url:
            visit(ref_url)

    return list(discovered.values())


def main():
    team_id_to_abbr = get_team_id_to_abbr()
    print(f"team_master teams={len(team_id_to_abbr)}")

    top_groups = fetch_json(GROUPS_URL)
    groups = discover_groups(top_groups)
    print(f"groups discovered={len(groups)}")

    # One best hierarchy row per team. Conference-marked memberships are
    # preferred over any fallback leaf-group membership.
    team_memberships = {}
    standings_rows = []
    seen_standings = set()

    for ref_url, group in groups:
        group_name = str(group.get("name", "")).strip()
        child_groups = get_child_groups(group)
        team_ids = get_group_team_ids(group)

        conf_name, conf_abbr, div_name, div_abbr, has_conference = hierarchy_labels(group, ref_url)

        # ESPN's CFB hierarchy marks true conferences with isConference.
        # If a leaf group has teams but no marked conference ancestor, keep
        # a low-priority fallback instead of silently losing those teams.
        use_fallback = bool(team_ids) and not has_conference and not child_groups

        if team_ids and (has_conference or use_fallback):
            if use_fallback:
                conf_name = group_name
                conf_abbr = str(group.get("abbreviation", "")).strip()
                div_name = ""
                div_abbr = ""

            membership_priority = 2 if has_conference else 1

            for team_id in team_ids:
                if team_id not in team_id_to_abbr:
                    continue

                existing = team_memberships.get(team_id)
                if existing and existing[0] > membership_priority:
                    continue

                team_memberships[team_id] = (
                    membership_priority,
                    {
                        "team_id": team_id,
                        "team_abbr": team_id_to_abbr.get(team_id, ""),
                        "conference": conf_name,
                        "conference_abbr": conf_abbr,
                        "division": div_name,
                        "division_abbr": div_abbr,
                        "season": SEASON,
                    },
                )

        standings_obj = group.get("standings", {})
        standings_ref = standings_obj.get("$ref") if isinstance(standings_obj, dict) else ""
        if standings_ref:
            for row in get_standings_rows(group, ref_url, team_id_to_abbr):
                key = tuple(str(row.get(column, "")) for column in STANDINGS_COLUMNS)
                if key in seen_standings:
                    continue
                seen_standings.add(key)
                standings_rows.append(row)

    master_rows = [entry[1] for entry in team_memberships.values()]

    master_rows.sort(
        key=lambda row: (
            row.get("conference", ""),
            row.get("division", ""),
            row.get("team_abbr", ""),
            row.get("team_id", ""),
        )
    )

    standings_rows.sort(
        key=lambda row: (
            row.get("conference", ""),
            row.get("division", ""),
            row.get("team_abbr", ""),
            row.get("standings_type", ""),
            row.get("stat_name", ""),
        )
    )

    missing_membership = sorted(set(team_id_to_abbr) - set(team_memberships))
    if missing_membership:
        print(f"WARNING: team_master teams with no conference membership={len(missing_membership)}")
        print(f"WARNING team_ids={','.join(missing_membership)}")

    os.makedirs(os.path.dirname(LEAGUE_MASTER_PATH), exist_ok=True)

    with open(LEAGUE_MASTER_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        writer.writeheader()
        writer.writerows(master_rows)
    print(f"rows={len(master_rows)} output={LEAGUE_MASTER_PATH}")

    with open(LEAGUE_STANDINGS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STANDINGS_COLUMNS)
        writer.writeheader()
        writer.writerows(standings_rows)
    print(f"rows={len(standings_rows)} output={LEAGUE_STANDINGS_PATH}")


if __name__ == "__main__":
    main()
