import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('NFL_DATA', os.path.join(ROOT, 'data'))
SITE = os.path.join(ROOT, 'site')
HERE = os.path.dirname(os.path.abspath(__file__))
import datetime as _dt
_t = _dt.date.today()
CUR = int(os.environ.get('NFL_SEASON', _t.year if _t.month >= 8 else _t.year - 1))
FIRST = CUR - 6   # seasons of history used
