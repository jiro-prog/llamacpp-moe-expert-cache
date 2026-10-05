# The routed-expert tensors of a split GGUF as CSV: shard path, layer, kind (0 gate, 1 up, 2 down), absolute data
# offset, tensor bytes, slice bytes (one expert), type, experts. Input of tools/densecopy.exe.
#   python ggufexps.py "<model dir>/<name>-*.gguf" [--gguf-py <llama.cpp>/gguf-py] > exps.csv
import argparse, glob, re, sys

ap = argparse.ArgumentParser()
ap.add_argument('pattern', help='glob of the GGUF shards')
ap.add_argument('--gguf-py', default=r'C:\llama-qwen\src\llama.cpp-a4cb4c61f\gguf-py', help="llama.cpp's gguf-py directory")
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
import gguf

files = sorted(glob.glob(a.pattern))
if not files:
    sys.exit('no file matches ' + a.pattern)
kinds = {'ffn_gate_exps': 0, 'ffn_up_exps': 1, 'ffn_down_exps': 2}
rows = []
for f in files:
    r = gguf.GGUFReader(f)
    for t in r.tensors:
        m = re.match(r'blk\.(\d+)\.(ffn_\w+_exps)\.weight$', t.name)
        if not m or m.group(2) not in kinds:
            continue
        ne = [int(x) for x in t.shape]
        rows.append((int(m.group(1)), kinds[m.group(2)], f, int(t.data_offset), int(t.n_bytes), int(t.n_bytes) // ne[-1], t.tensor_type.name, ne[-1]))
rows.sort()
print('layer,kind,path,offset,bytes,slice,type,n_expert')
for L, k, f, off, nb, sl, ty, ne in rows:
    print(f'{L},{k},{f},{off},{nb},{sl},{ty},{ne}')
