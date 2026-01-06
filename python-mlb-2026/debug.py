import json, requests

def debug_pitcher_career(pid: int):
    url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    params = {"stats": "career", "group": "pitching"}  # <-- no gameType for debugging
    r = requests.get(url, params=params, timeout=30)
    print("URL:", r.url)
    print("status:", r.status_code)
    data = r.json()
    print("top keys:", list(data.keys()))
    stats = data.get("stats") or []
    print("stats blocks:", len(stats))
    if stats:
        print("stats[0] keys:", list(stats[0].keys()))
        splits = stats[0].get("splits") or []
        print("splits:", len(splits))
        if splits:
            print("split[0] keys:", list(splits[0].keys()))
            stat = splits[0].get("stat") or {}
            # print a few common fields if present
            for k in ["inningsPitched","hits","baseOnBalls","strikeOuts","homeRuns","era","whip","battersFaced"]:
                if k in stat:
                    print(k, "=>", stat[k])
            # show full stat dict size
            print("stat keys count:", len(stat.keys()))
    # uncomment if you want full payload:
    # print(json.dumps(data, indent=2)[:2000])

debug_pitcher_career(678394)
