# Expert-cache simulator over a routing trace (expert-prefetch LLAMA_EXPERT_PREFETCH_TRACE format:
# "<pass> <layer> <rows> <experts...>"). Decode lines only (rows == 1); one unit = one (layer, expert) with its
# gate/up/down slices (~3.13 MB). Prints the hit rate per policy and cache size.
#   python cachesim.py <trace> [unit_MB=3.13] [sizes_GB=4,6,7,8,9,10,12]
import sys, collections

def load(path):
    acc = []   # list of passes; each pass = list of (layer, [experts])
    cur, cur_pass = [], None
    for line in open(path):
        f = line.split()
        if len(f) < 4 or f[2] != '1':
            continue
        p, il, ex = int(f[0]), int(f[1]), [int(x) for x in f[3:]]
        if p != cur_pass and cur:
            acc.append(cur)
            cur = []
        cur_pass = p
        cur.append((il, ex))
    if cur:
        acc.append(cur)
    return acc

def seq(passes):
    for ps in passes:
        for il, ex in ps:
            for e in ex:
                yield (il, e)

def lru(passes, cap):
    c = collections.OrderedDict(); hit = n = 0
    for k in seq(passes):
        n += 1
        if k in c:
            hit += 1; c.move_to_end(k)
        else:
            c[k] = 1
            if len(c) > cap:
                c.popitem(last=False)
    return hit / n

def lfu(passes, cap, decay=0.0):
    # evict the unit with the lowest (decayed) access count; ties -> least recent
    import heapq
    cnt = collections.defaultdict(float); inc = set(); last = {}
    t = 0; hit = n = 0; scale = 1.0
    heap = []
    for k in seq(passes):
        n += 1; t += 1
        if decay:
            scale *= (1 + decay)
        cnt[k] += scale
        last[k] = t
        if k in inc:
            hit += 1
        else:
            inc.add(k)
            if len(inc) > cap:
                # lazy: rebuild candidates occasionally
                victim = min(inc - {k}, key=lambda u: (cnt[u], last[u]))
                inc.discard(victim)
        heapq  # unused
    return hit / n

def lfu_fast(passes, cap, half_life_tokens=0):
    # exact LFU with optional exponential decay, using a lazy heap of (score, last, unit)
    import heapq
    score = collections.defaultdict(float); last = {}
    inc = set(); heap = []; t = 0; hit = n = 0
    g = 1.0
    per_tok = 2 ** (1.0 / half_life_tokens) if half_life_tokens else 1.0
    tok = 0
    for ps in passes:
        tok += 1
        if half_life_tokens:
            g *= per_tok
        for il, ex in ps:
            for e in ex:
                k = (il, e); n += 1; t += 1
                score[k] += g; last[k] = t
                if k in inc:
                    hit += 1
                else:
                    inc.add(k)
                heapq.heappush(heap, (score[k], last[k], k))
                keep = []
                while len(inc) > cap:
                    s, l, u = heapq.heappop(heap)
                    if u not in inc or s != score[u] or l != last[u]:
                        continue   # stale entry
                    if u == k:
                        keep.append((s, l, u))
                        continue
                    inc.discard(u)
                for x in keep:
                    heapq.heappush(heap, x)
    return hit / n

def belady(passes, cap):
    keys = list(seq(passes))
    nxt = [0] * len(keys); seen = {}
    for i in range(len(keys) - 1, -1, -1):
        nxt[i] = seen.get(keys[i], 1 << 60); seen[keys[i]] = i
    import heapq
    inc = {}; heap = []; hit = 0
    for i, k in enumerate(keys):
        if k in inc:
            hit += 1
        inc[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
        while len(inc) > cap:
            nn, u = heapq.heappop(heap)
            if u in inc and inc[u] == -nn:
                del inc[u]
    return hit / len(keys)

if __name__ == '__main__':
    path = sys.argv[1]
    unit = float(sys.argv[2]) if len(sys.argv) > 2 else 3.13
    sizes = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '4,6,7,8,9,10,12').split(',')]
    passes = load(path)
    ntok = len(passes)
    uniq = len(set(seq(passes)))
    print(f'{ntok} decode tokens, {sum(len(p) for p in passes)} layer steps, {uniq} distinct units ({uniq * unit / 1024:.1f} GB)')
    print('size_GB  units   LRU    LFU   LFU(hl=200) LFU(hl=50) Belady')
    for gb in sizes:
        cap = int(gb * 1024 / unit)
        r = [lru(passes, cap), lfu_fast(passes, cap), lfu_fast(passes, cap, 200), lfu_fast(passes, cap, 50), belady(passes, cap)]
        print(f'{gb:6.1f} {cap:6d}  ' + '  '.join(f'{x * 100:5.1f}%' for x in r))

def lru_per_layer(passes, cap_total, n_layers=48):
    caps = cap_total // n_layers
    cs = collections.defaultdict(collections.OrderedDict); hit = n = 0
    for il, e in seq(passes):
        c = cs[il]; n += 1
        if e in c:
            hit += 1; c.move_to_end(e)
        else:
            c[e] = 1
            if len(c) > caps:
                c.popitem(last=False)
    return hit / n

def slru(passes, cap, prot_frac=0.8):
    prot_cap = int(cap * prot_frac)
    prob = collections.OrderedDict(); prot = collections.OrderedDict(); hit = n = 0
    for k in seq(passes):
        n += 1
        if k in prot:
            hit += 1; prot.move_to_end(k)
        elif k in prob:
            hit += 1; del prob[k]; prot[k] = 1
            if len(prot) > prot_cap:
                d, _ = prot.popitem(last=False); prob[d] = 1
        else:
            prob[k] = 1
        while len(prob) + len(prot) > cap:
            if prob:
                prob.popitem(last=False)
            else:
                prot.popitem(last=False)
    return hit / n

def arc(passes, c):
    T1 = collections.OrderedDict(); T2 = collections.OrderedDict()
    B1 = collections.OrderedDict(); B2 = collections.OrderedDict()
    p = 0; hit = n = 0
    def replace(x):
        nonlocal p
        if T1 and (len(T1) > p or (x in B2 and len(T1) == p)):
            k, _ = T1.popitem(last=False); B1[k] = 1
        else:
            k, _ = T2.popitem(last=False); B2[k] = 1
    for x in seq(passes):
        n += 1
        if x in T1:
            hit += 1; del T1[x]; T2[x] = 1
        elif x in T2:
            hit += 1; T2.move_to_end(x)
        elif x in B1:
            p = min(c, p + max(len(B2) // max(1, len(B1)), 1))
            replace(x); del B1[x]; T2[x] = 1
        elif x in B2:
            p = max(0, p - max(len(B1) // max(1, len(B2)), 1))
            replace(x); del B2[x]; T2[x] = 1
        else:
            if len(T1) + len(B1) == c:
                if len(T1) < c:
                    B1.popitem(last=False); replace(x)
                else:
                    T1.popitem(last=False)
            elif len(T1) + len(B1) < c and len(T1) + len(T2) + len(B1) + len(B2) >= c:
                if len(T1) + len(T2) + len(B1) + len(B2) == 2 * c:
                    B2.popitem(last=False)
                replace(x)
            T1[x] = 1
    return hit / n
