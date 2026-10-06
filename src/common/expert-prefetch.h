#pragma once

// Expert prefetch for MoE models whose routed experts stay memory-mapped on the CPU side (-ot exps=CPU).
//
// Without it, every expert slice that is not in RAM is read by page faults on the compute threads, 32-64 KB at a
// time (Windows reads mapped files in such small clusters), so the SSD sits at a fraction of its speed.
// With it, the eval callback catches each layer's routing result ("ffn_moe_topk-N"), and a pool of worker threads
// reads exactly the selected expert slices with PrefetchVirtualMemory (large, parallel reads) and touches them,
// so the compute threads find the pages resident. Slices of layers that are done are dropped from the working
// set (VirtualUnlock): they stay in the standby list as file cache, which keeps Available RAM high without a hard
// working-set cap.
//
// Enabled with LLAMA_EXPERT_PREFETCH=1. Options (environment):
//   LLAMA_EXPERT_PREFETCH_WORKERS  worker threads (default 6)
//   LLAMA_EXPERT_PREFETCH_CHUNK_KB size of one prefetch job (default 2048)
//   LLAMA_EXPERT_PREFETCH_TOUCH    1 = touch the pages after prefetching them (default 1)
//   LLAMA_EXPERT_PREFETCH_EVICT    drop slices of layers this many layers back from the working set (0 = off, default 2)
//   LLAMA_EXPERT_PREFETCH_STATS    1 = print a line per pass (default 0)
//   LLAMA_EXPERT_PREFETCH_TIMING   1 = also catch the MoE end ("ffn_moe_down-N", the last CPU node; LLAMA_EXPERT_PREFETCH_END_OUT=1:
//                                  the GPU node "ffn_moe_out-N", one more GPU sync per layer) and split each pass into CPU-MoE / fetch / GPU time
//                                  (adds one callback per layer; default 0)
//   LLAMA_EXPERT_PREFETCH_DIAG     1 = also a "diag" line per pass: pages already in the working set, disk reads,
//                                  page faults, per-page fault-in cost, CPU times (slows the fetch a little; default 0)
//   LLAMA_EXPERT_PREFETCH_TRACE    file to append the routing to, one line per layer and pass (default none)
//   LLAMA_GPU_KEEPALIVE            spin one warp for this many us at a time while the CPU computes a layer's experts,
//                                  so the driver keeps the GPU in a fast P-state during decode (0 = off, default)
//   LLAMA_EXPERT_CACHE_GB          > 0: decode reads the experts from a cache of this size in locked memory, filled by
//                                  unbuffered reads of LLAMA_EXPERT_CACHE_FILE(.bin/.index, tools\densecopy) instead of the
//                                  mapped GGUF (needs the ggml-cpu expert-data hook); LLAMA_EXPERT_CACHE_MMAP_GB = working
//                                  set left for everything else (default 2)

#include "ggml-backend.h"

// returns the callback to install as cb_eval (nullptr when disabled); user_data goes to cb_eval_user_data
ggml_backend_sched_eval_callback expert_prefetch_callback(void ** user_data);
