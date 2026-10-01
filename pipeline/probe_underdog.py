"""One-off: save the raw Underdog (us_dfs) response for one upcoming game, to learn its price format."""
import json, os, urllib.request, urllib.parse, datetime
BASE = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
KEY = os.environ['ODDS_API_KEY']
def get(path, **p):
    p.update(apiKey=KEY, oddsFormat='american')
    with urllib.request.urlopen(f'{BASE}{path}?{urllib.parse.urlencode(p)}', timeout=30) as r:
        return json.load(r), dict(r.headers)
events, h = get('/events')
now = datetime.datetime.now(datetime.timezone.utc)
ev = sorted((e for e in events if datetime.datetime.fromisoformat(e['commence_time'].replace('Z', '+00:00')) > now),
            key=lambda e: e['commence_time'])[0]
d, h = get(f"/events/{ev['id']}/odds", regions='us_dfs',
           markets='player_rush_yds,player_reception_yds,player_rush_yds_alternate,player_reception_yds_alternate')
out = {'event': ev, 'response': d, 'credits_left': h.get('x-requests-remaining') or h.get('X-Requests-Remaining'),
       'credits_used': h.get('x-requests-used') or h.get('X-Requests-Used'), 'last_cost': h.get('x-requests-last') or h.get('X-Requests-Last')}
os.makedirs('data', exist_ok=True)
json.dump(out, open('data/probe_underdog.json', 'w'), indent=1)
print('saved; credits left', out['credits_left'], 'last cost', out['last_cost'])
