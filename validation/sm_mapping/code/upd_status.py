# Small helper: merge updates into STATUS.json (sm-blind, R16 V1)
import sys, json, os, datetime, argparse
sys.stdout.reconfigure(encoding='utf-8')
MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
P = os.path.join(MY, 'STATUS.json')
ap = argparse.ArgumentParser()
ap.add_argument('--done', action='append', default=[])
ap.add_argument('--file', action='append', default=[])
ap.add_argument('--key', action='append', default=[], help='name=value (value parsed as JSON if possible)')
ap.add_argument('--next', default=None)
ap.add_argument('--remaining', default=None, help='items separated by |')
ap.add_argument('--complete', action='store_true')
a = ap.parse_args()
st = json.load(open(P, encoding='utf-8'))
for d in a.done:
    if d not in st['done']:
        st['done'].append(d)
for f in a.file:
    if f not in st['files']:
        st['files'].append(f)
for kv in a.key:
    k, v = kv.split('=', 1)
    try:
        v = json.loads(v)
    except Exception:
        pass
    st['key_numbers'][k] = v
if a.next is not None:
    st['next_command'] = a.next
if a.remaining is not None:
    st['remaining'] = [x for x in a.remaining.split('|') if x]
st['complete'] = bool(a.complete) or st.get('complete', False) if a.complete else False
st['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
json.dump(st, open(P, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('STATUS updated', st['updated_utc'])
