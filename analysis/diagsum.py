# Summarize the "expert-prefetch: pass" + "diag pass" lines of server logs (decode passes only: rows 1).
#   python diagsum.py <server log> [<server log> ...]
import re, sys, statistics as st

PASS = re.compile(r'pass ([\d.]+) s \| rows (\d+) .*?moe ([\d.]+) s, fetch ([\d.]+) s, gpu ([\d.]+) s')
DIAG = {
    'ws0':   r'in WS ([\d.]+)% before',
    'diskMB': r'disk (\d+) MB',
    'reads': r'MB in (\d+) reads',
    'rdKB':  r'avg (\d+) KB',
    'rdus':  r'KB, (\d+) us each',
    'QD':    r'QD ([\d.]+)',
    'faults': r'faults (\d+)',
    'pfcold': r'cold jobs \d+: prefetch (\d+) us',
    'us/pg': r'= ([\d.]+) us/page',
    'coldk': r'us/page \(([\d.]+)k pages\)',
    'wait':  r'queue wait (\d+) us/job',
    'kern':  r'cpu kernel (\d+) ms',
    'user':  r'user (\d+) ms',
    'query': r'query ([\d.]+) s',
}

def summarize(path):
    rows = []
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    for i, l in enumerate(lines):
        m = PASS.search(l)
        if not m or m.group(2) != '1':
            continue
        r = dict(pass_s=float(m.group(1)), moe=float(m.group(3)), fetch=float(m.group(4)), gpu=float(m.group(5)))
        if i + 1 < len(lines) and 'diag pass' in lines[i + 1]:
            d = lines[i + 1]
            for k, rx in DIAG.items():
                mm = re.search(rx, d)
                if mm:
                    r[k] = float(mm.group(1))
        rows.append(r)
    # skip passes longer than 2 s (prompt tails, first token)
    rows = [r for r in rows if r['pass_s'] < 2]
    keys = ['pass_s', 'moe', 'fetch', 'gpu'] + list(DIAG)
    out = {'n': len(rows)}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if v:
            out[k] = st.mean(v)
    return out

if __name__ == '__main__':
    keys = None
    for p in sys.argv[1:]:
        s = summarize(p)
        if keys is None:
            keys = [k for k in s]
            print('log'.ljust(28) + ''.join(k.rjust(8) for k in keys))
        print(p.split('\\')[-1].split('/')[-1][:27].ljust(28) + ''.join(
            (('%8.3f' if s.get(k, 0) < 10 else '%8.0f') % s.get(k, 0)) for k in keys))
