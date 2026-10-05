// Copy the routed experts of the Qwen GGUF shards into one never-mapped file laid out per (layer, expert):
//   [gate slice][up slice][down slice]   (each slice is a multiple of 4 KiB, so everything stays 4 KiB aligned)
// Unbuffered reads and writes, so the new file never gets a cache map or a data section: non-buffered reads of a file
// with a live section are served one at a time (found with Kimi K3, 2026-09-26), and llama.cpp maps the shards.
//   densecopy.exe <exps.csv from bench\ggufexps.py> <out prefix>   -> <prefix>.bin, <prefix>.index
// The index has one line per layer: "layer base unit gate up down n_expert" (bytes; base = offset of expert 0).
// At the end a few random units are read back and compared with the source.
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <random>

struct row { int layer, kind; std::string path; int64_t off, bytes, slice; std::string type; int n_expert; };

static HANDLE open_read(const std::string & p) {
    HANDLE h = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING, nullptr);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open %s failed %lu\n", p.c_str(), GetLastError()); exit(1); }
    return h;
}

// unbuffered read of [off, off + n) into dst (any alignment of off/n): reads the aligned superset through a bounce
static void read_range(HANDLE h, int64_t off, int64_t n, char * dst, char * bounce, int64_t bounce_size) {
    int64_t done = 0;
    while (done < n) {
        const int64_t a   = (off + done) & ~(int64_t) 4095;
        const int64_t lead = off + done - a;
        const int64_t want = std::min<int64_t>(bounce_size, (lead + (n - done) + 4095) & ~(int64_t) 4095);
        OVERLAPPED ov{};
        ov.Offset     = (DWORD) (a & 0xFFFFFFFF);
        ov.OffsetHigh = (DWORD) (a >> 32);
        DWORD got = 0;
        if (!ReadFile(h, bounce, (DWORD) want, &got, &ov)) {
            // synchronous handle: ReadFile returns when done
            fprintf(stderr, "read failed %lu at %lld\n", GetLastError(), (long long) a);
            exit(1);
        }
        const int64_t use = std::min<int64_t>((int64_t) got - lead, n - done);
        if (use <= 0) { fprintf(stderr, "short read at %lld\n", (long long) a); exit(1); }
        memcpy(dst + done, bounce + lead, (size_t) use);
        done += use;
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: densecopy exps.csv out_prefix\n"); return 1; }
    std::vector<row> rows;
    {
        FILE * f = fopen(argv[1], "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        char line[4096];
        fgets(line, sizeof line, f);   // header
        while (fgets(line, sizeof line, f)) {
            row r;
            char path[2048], type[32];
            long long off, bytes, slice;
            if (sscanf(line, "%d,%d,%2047[^,],%lld,%lld,%lld,%31[^,],%d", &r.layer, &r.kind, path, &off, &bytes, &slice, type, &r.n_expert) != 8) continue;
            r.path = path; r.off = off; r.bytes = bytes; r.slice = slice; r.type = type;
            rows.push_back(r);
        }
        fclose(f);
    }
    std::map<int, row[3]> layers;
    for (auto & r : rows) layers[r.layer][r.kind] = r;
    const std::string out_bin = std::string(argv[2]) + ".bin", out_idx = std::string(argv[2]) + ".index";

    std::map<std::string, HANDLE> src;
    for (auto & r : rows) if (!src.count(r.path)) src[r.path] = open_read(r.path);

    HANDLE dst = CreateFileA(out_bin.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (dst == INVALID_HANDLE_VALUE) { fprintf(stderr, "create %s failed %lu\n", out_bin.c_str(), GetLastError()); return 1; }

    const int64_t bounce_size = 64ll << 20;
    char * bounce = (char *) VirtualAlloc(nullptr, bounce_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    int64_t max_tensor = 0;
    for (auto & r : rows) max_tensor = std::max(max_tensor, r.bytes);
    char * in[3];
    for (int k = 0; k < 3; ++k) in[k] = (char *) VirtualAlloc(nullptr, max_tensor, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    const int batch = 32;   // units per write
    int64_t max_unit = 0;
    for (auto & kv : layers) max_unit = std::max(max_unit, kv.second[0].slice + kv.second[1].slice + kv.second[2].slice);
    char * outbuf = (char *) VirtualAlloc(nullptr, max_unit * batch, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!bounce || !in[0] || !in[1] || !in[2] || !outbuf) { fprintf(stderr, "out of memory\n"); return 1; }

    FILE * idx = fopen(out_idx.c_str(), "w");
    int64_t pos = 0;
    const ULONGLONG t0 = GetTickCount64();
    ULONGLONG last = t0;
    for (auto & kv : layers) {
        const int L = kv.first;
        row * r = kv.second;
        const int ne = r[0].n_expert;
        for (int k = 0; k < 3; ++k) {
            if (r[k].slice % 4096 || r[k].n_expert != ne || r[k].slice * ne != r[k].bytes) {
                fprintf(stderr, "layer %d kind %d: unexpected sizes\n", L, k); return 1;
            }
            read_range(src[r[k].path], r[k].off, r[k].bytes, in[k], bounce, bounce_size);
        }
        const int64_t unit = r[0].slice + r[1].slice + r[2].slice;
        fprintf(idx, "%d %lld %lld %lld %lld %lld %d %s %s %s\n", L, (long long) pos, (long long) unit, (long long) r[0].slice,
                (long long) r[1].slice, (long long) r[2].slice, ne, r[0].type.c_str(), r[1].type.c_str(), r[2].type.c_str());
        for (int e0 = 0; e0 < ne; e0 += batch) {
            const int n = std::min(batch, ne - e0);
            char * o = outbuf;
            for (int e = e0; e < e0 + n; ++e) {
                for (int k = 0; k < 3; ++k) {
                    memcpy(o, in[k] + (int64_t) e * r[k].slice, (size_t) r[k].slice);
                    o += r[k].slice;
                }
            }
            OVERLAPPED ov{};
            ov.Offset = (DWORD) (pos & 0xFFFFFFFF);
            ov.OffsetHigh = (DWORD) (pos >> 32);
            DWORD put = 0;
            if (!WriteFile(dst, outbuf, (DWORD) (o - outbuf), &put, &ov) || put != (DWORD) (o - outbuf)) {
                fprintf(stderr, "write failed %lu\n", GetLastError()); return 1;
            }
            pos += o - outbuf;
        }
        const ULONGLONG now = GetTickCount64();
        if (now - last > 5000 || L == layers.rbegin()->first) {
            last = now;
            fprintf(stderr, "layer %d done, %.1f GB, %.2f GB/s\n", L, pos / 1e9, pos / 1e9 / ((now - t0) / 1000.0));
        }
    }
    fclose(idx);
    CloseHandle(dst);

    // read back random units and compare with the source
    HANDLE chk = open_read(out_bin);
    std::mt19937 rng(12345);
    int bad = 0, n_chk = 0;
    int64_t base = 0;
    std::map<int, int64_t> bases;
    for (auto & kv : layers) { bases[kv.first] = base; row * r = kv.second; base += (r[0].slice + r[1].slice + r[2].slice) * r[0].n_expert; }
    char * a = (char *) VirtualAlloc(nullptr, max_unit, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    char * b = (char *) VirtualAlloc(nullptr, max_unit, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (int i = 0; i < 200; ++i) {
        auto it = layers.begin();
        std::advance(it, rng() % layers.size());
        row * r = it->second;
        const int e = (int) (rng() % r[0].n_expert);
        const int64_t unit = r[0].slice + r[1].slice + r[2].slice;
        read_range(chk, bases[it->first] + e * unit, unit, a, bounce, bounce_size);
        int64_t o = 0;
        for (int k = 0; k < 3; ++k) {
            read_range(src[r[k].path], r[k].off + e * r[k].slice, r[k].slice, b + o, bounce, bounce_size);
            o += r[k].slice;
        }
        bad += memcmp(a, b, (size_t) unit) != 0;
        n_chk++;
    }
    CloseHandle(chk);
    fprintf(stderr, "DONE %.1f GB in %.0f s; check: %d of %d random units differ\n", pos / 1e9, (GetTickCount64() - t0) / 1000.0, bad, n_chk);
    return bad ? 2 : 0;
}
