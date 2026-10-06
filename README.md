# llamacpp-moe-expert-cache

**[日本語](#概要) | [English](#overview)**

---

## 概要

総 125B パラメータの MoE モデル（Qwen3.8-Flash-Next、UD-Q4_K_XL で 111GB）を、**RAM 16GB・VRAM 8GB の Windows PC** で
llama.cpp を使って動かすための改造です。RAM に収まらない routed expert（77GB）を SSD から読みながら生成します。

| | 配布版（mmap のみ） | 初期の改造（expert 先読み） | 現在 |
|---|---|---|---|
| 生成速度 | 1.4〜2.4 tok/s | 2.7〜3.9 tok/s | **6.5〜8.0 tok/s** |
| 日本語 10 問（約 3600 トークン生成）の所要 | – | 1243 s | **565 s** |
| 24〜28 トークンのプロンプト処理 | – | 3.5〜4.4 t/s | **8.1〜8.5 t/s** |

デコードの改造（キャッシュ・先読み・キープアライブなど）では出力は 1 文字も変わりません（温度 0 の 3 問、温度 0.7・シード固定の 10 問で確認）。
32〜1279 トークンのプロンプトは GPU ではなく CPU で計算するようになったため、その後の回答が以前と変わることがあります（数値の差）。
測定と設計の詳細は [docs/findings.md](docs/findings.md) にあります。

### 仕組み

- **自前の expert キャッシュ**: デコード時の expert を、ロックした私有メモリの LRU キャッシュ（9GB）から読みます。短いプロンプト（1280 トークン未満）も同じ経路で読みます。
  外れたときは、expert だけを並べ直したコピー（一度もマップしないファイル）からバッファなしの `ReadFile` で 1MB 前後ずつ読みます。
  mmap のページフォールト（1 トークン約 9 万回）と、細切れの SSD 読み込みをなくしました。
- **ggml-cpu への差し替え口**: MUL_MAT_ID が expert のデータの在りかを問い合わせる関数と、準備済みかを問い合わせる関数を
  足しました。計算スレッドは読み込み済みの expert から先に計算し、読み込み中のものは待ちます。
- **次の層の先読み**: 層 N の FFN 入力に層 N+1 のルーターを掛けるノードをグラフに足し、予測した上位 8 個の expert を
  SSD が空いている間に読みます。
- **GPU キープアライブ**: デコード中に GPU が省電力状態（メモリ 810MHz）に落ちるのを、CPU が expert を計算している
  隙間だけ 1 ワープを空回りさせて防ぎます（Windows の WDDM ではストリームを分けてもカーネルが並行しないため、隙間だけ）。
- **GPU の同期を減らす**: ggml は観測したノードの直後に GPU を同期するので、各層の区切りを GPU のノードから CPU 側の
  最後のノードに移しました（1 層 1 回の同期が減って +5%）。
- **プロンプトの経路の選び方**: GPU 経路は 1 バッチごとに全 expert（約 75GB）を GPU に送るので、1280 トークン未満は
  CPU＋キャッシュで計算します。それ以上を GPU で処理する間は、キャッシュの冷たい部分 3.5GB を mmap 側に回します。

### 必要環境

- Windows 10/11、NVIDIA GPU（RTX 3060 Ti 8GB で確認）、RAM 16GB 以上、NVMe SSD（モデル 111GB＋expert のコピー 77GB）
- [w64devkit](https://github.com/skeeto/w64devkit)（GCC）、CMake、Ninja、Python 3（`gguf` パッケージは llama.cpp 同梱のものを使用）、git

### セットアップ

```powershell
# 1. llama.cpp b11361 のソース取得とパッチ適用、同じ版の CUDA 配布バイナリの取得（C:\llama-qwen に展開）
.\scripts\setup.ps1
# 2. ビルド（gcc / cmake / ninja を PATH に置くか、TOOLS_PATH に bin ディレクトリを ; 区切りで指定）
C:\llama-qwen\build.cmd
# 3. モデルの取得
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --local-dir C:\models\Qwen3.8-Flash-Next
# 4. expert のコピー（約 3 分、77GB）
python tools\ggufexps.py "C:\models\Qwen3.8-Flash-Next\UD-Q4_K_XL\*.gguf" > exps.csv
g++ -O2 -std=c++17 -o tools\densecopy.exe tools\densecopy.cpp
tools\densecopy.exe exps.csv C:\models\Qwen3.8-Flash-Next\UD-Q4_K_XL\experts-dense
# 5. 起動（127.0.0.1:8091、OpenAI 互換 API）
C:\llama-qwen\qwen-run.ps1
```

### 主なオプション（`qwen-run.ps1`）

| 引数 | 既定 | 内容 |
|---|---|---|
| `-CacheGB` | 8.5 | expert キャッシュの大きさ GiB（0 で無効、従来の mmap） |
| `-CacheMmapGB` | 0.6 | デコード中にキャッシュ以外に残すワーキングセット |
| `-PrefillReleaseGB` | 3.5 | 長いプロンプトを GPU で処理する間だけ mmap 側に回すキャッシュ（その後デコードの前に戻す） |
| `-OffloadMinBatch` | 1280 | この長さ以上のプロンプトのバッチは GPU（全 expert を送る）、未満は CPU＋キャッシュ |
| `-Predict` / `-PredictWorkers` | 8 / 2 | 次の層の先読みの個数と、先読みに使うワーカー数の上限（-1 で無効） |
| `-KeepAlive` | 300 | GPU キープアライブの空回り 1 本の長さ µs（0 で無効） |
| `-Stats -Timing` | – | 1 トークンごとの内訳を 1 秒ごとにログへ書く |

### 構成

```
patches/llama.cpp-b11361.patch   llama.cpp への変更一式（新規ファイルを含む）
src/common/expert-prefetch.*     キャッシュ・先読み・キープアライブ本体（パッチに含まれるものと同じ）
tools/                           expert のコピー（densecopy.cpp）、GGUF の expert 一覧（ggufexps.py）
analysis/                        キャッシュ方式のシミュレーション（cachesim.py）、診断ログの集計（diagsum.py）
scripts/                         セットアップ、ビルド、起動、LAN 公開用のファイアウォール設定
bench/                           A/B 測定（3 問・10 問・長文プロンプト、ルーティングの記録）
docs/findings.md                 測定の記録
```

### 注意

- 1280 トークン以上のプロンプトは GPU で処理し、77 → 65.5 t/s（キャッシュに RAM を割いた分。処理中はキャッシュの一部を手放して補っています）。
- Windows 専用です（`PrefetchVirtualMemory`、`WaitOnAddress`、バッファなし I/O、nvcuda.dll を直接使います）。
- 特定のモデル（`qwen4exp` アーキテクチャ）で測っています。他の MoE でも動く作りですが、未確認です。

---

## Overview

Patches and tools to run a 125B-parameter MoE (Qwen3.8-Flash-Next, 111 GB as UD-Q4_K_XL) with llama.cpp on a
**Windows PC with 16 GB RAM and an 8 GB GPU**, streaming the routed experts (77 GB) from an NVMe SSD.

| | stock (mmap only) | first version (expert prefetch) | now |
|---|---|---|---|
| decode | 1.4-2.4 tok/s | 2.7-3.9 tok/s | **6.5-8.0 tok/s** |
| 10 Japanese prompts (~3600 generated tokens) | - | 1243 s | **565 s** |

The decode changes keep the output token for token (3 prompts at temperature 0, 10 prompts at temperature 0.7 with a
fixed seed). Prompts of 32-1279 tokens are now computed on the CPU instead of the GPU, so later text can differ slightly.
Measurements and design notes (Japanese): [docs/findings.md](docs/findings.md).

### How it works

- **Expert cache**: decode and prompts under 1280 tokens read experts from an LRU cache in locked private memory (9 GB). A miss is one unbuffered
  `ReadFile` per slice (about 1 MB) from a copy of the experts that is never memory-mapped (on Windows, unbuffered reads
  of a file with a live section are served one at a time). This replaces ~90k page faults per token and fragmented reads.
- **ggml-cpu hooks**: MUL_MAT_ID asks where an expert's data is and whether it is ready; compute threads do the
  experts that are already in memory first and wait for the rest.
- **Next-layer read-ahead**: a graph node applies layer N+1's router to layer N's FFN input; the 8 best predicted
  experts are read while the SSD would otherwise be idle.
- **GPU keep-alive**: during decode the driver drops the GPU to P5 (810 MHz memory clock). One spinning warp, launched
  only while the CPU computes experts and the GPU is idle (WDDM does not run kernels of different streams concurrently),
  keeps it in P2.
- **Fewer GPU syncs**: ggml synchronizes the GPU after every observed node, so each layer's boundary is observed at the
  last CPU node instead of a GPU node (+5%).
- **Prompt path**: the GPU path copies every expert (~75 GB) per batch, so prompts under 1280 tokens run on the CPU
  through the cache; while a longer one runs on the GPU, 3.5 GB of the cache is handed to the memory-mapped side.

### Requirements

Windows 10/11, an NVIDIA GPU (tested: RTX 3060 Ti 8 GB), 16 GB RAM, an NVMe SSD (111 GB model + 77 GB expert copy),
w64devkit (GCC), CMake, Ninja, Python 3, git.

### Setup

See the commands in the Japanese section above: `scripts\setup.ps1` (llama.cpp b11361 source + patch + CUDA release
binaries into `C:\llama-qwen`), `build.cmd`, download the GGUF, build the expert copy with `tools\ggufexps.py` and
`tools\densecopy.exe`, then `qwen-run.ps1` (OpenAI-compatible API on 127.0.0.1:8091).

### Notes

- Prompts of 1280+ tokens run on the GPU at 65.5 tok/s (77 before the cache; part of the cache is handed back meanwhile).
- Windows only. Measured with the `qwen4exp` architecture only.

## License

MIT (see [LICENSE](LICENSE)). The patch modifies [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, the ggml authors).
