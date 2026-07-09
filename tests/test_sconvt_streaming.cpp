// Streaming SConvTranspose1d parity: chunked (with StreamingCache) must equal
// single-shot, bit-exact modulo fp noise.
//
// This test is fully SELF-CONTAINED: it generates random kernel + bias + input
// tensors in-memory (no fixture/model load), so it always runs in CI. The
// invariant is streaming-vs-single-shot through the SAME kernel via the SAME
// functions (vv::sconv_transpose1d_causal vs ..._streaming) — any consistent
// random kernel exercises it; there is no PyTorch/reference comparison.
#include "conv1d.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// Single-shot: run vv::sconv_transpose1d_causal on the whole input.
bool run_single(struct ggml_tensor* w, struct ggml_tensor* b,
                const std::vector<float>& in, int T, int C_in, int stride,
                std::vector<float>* out, int* T_out, int* C_out) {
    struct ggml_init_params p{}; p.mem_size = 64ull<<20; p.no_alloc = false;
    struct ggml_context* ctx = ggml_init(p);
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, C_in, 1);
    std::memcpy(x->data, in.data(), sizeof(float)*in.size());
    struct ggml_tensor* y = vv::sconv_transpose1d_causal(ctx, x, w, b, stride);
    struct ggml_cgraph* gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, y);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return false; }
    *T_out = (int)y->ne[0]; *C_out = (int)y->ne[1];
    out->assign((size_t)y->ne[0]*y->ne[1], 0.f);
    std::memcpy(out->data(), y->data, sizeof(float)*out->size());
    ggml_free(ctx); return true;
}

// One streaming chunk (mirrors test_encoder_chunked_parity per-chunk block).
// On success, *out holds this chunk's emitted samples in [emit_len, C_out]
// (time-fastest) layout; *T_out=emit_len, *C_out=C_out.
bool run_stream_chunk(struct ggml_tensor* w, struct ggml_tensor* b,
                      const std::vector<float>& seg, int seg_T, int C_in, int stride,
                      vv::StreamingCache& cache, std::vector<float>* out, int* T_out, int* C_out) {
    struct ggml_init_params p{}; p.mem_size = ggml_tensor_overhead()*8192 + ggml_graph_overhead_custom(8192,false); p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seg_T, C_in, 1);
    struct ggml_tensor* y = vv::sconv_transpose1d_causal_streaming(ctx, x, w, b, stride, cache, "convt");
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false); ggml_build_forward_expand(gf, y);
    for (auto& kv : cache) if (kv.second.next_view) ggml_build_forward_expand(gf, kv.second.next_view);
    ggml_backend_buffer_t buf = vv::allocate_ctx_tensors(ctx);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(x, seg.data(), 0, sizeof(float)*seg.size());
    for (auto& kv : cache) {
        vv::StreamingCacheEntry& e = kv.second; if (!e.prefix || e.T==0) continue;
        const size_t need = (size_t)e.T*e.C;
        if (cache.is_first_chunk || e.data.size()!=need) { std::vector<float> z(need,0.f); ggml_backend_tensor_set(e.prefix, z.data(), 0, sizeof(float)*need); }
        else ggml_backend_tensor_set(e.prefix, e.data.data(), 0, sizeof(float)*need);
    }
    if (!vv::compute_graph(gf)) { ggml_backend_buffer_free(buf); ggml_free(ctx); return false; }
    *T_out = (int)y->ne[0]; *C_out = (int)y->ne[1];
    const size_t n = (size_t)y->ne[0]*y->ne[1];
    out->assign(n, 0.f);
    ggml_backend_tensor_get(y, out->data(), 0, sizeof(float)*n);
    for (auto& kv : cache) {
        vv::StreamingCacheEntry& e = kv.second; if (!e.next_view || e.T==0) continue;
        const size_t nn = (size_t)e.T*e.C; e.data.assign(nn,0.f);
        ggml_backend_tensor_get(e.next_view, e.data.data(), 0, sizeof(float)*nn);
        e.next_view=nullptr; e.prefix=nullptr;
    }
    cache.is_first_chunk=false;
    ggml_backend_buffer_free(buf); ggml_free(ctx); return true;
}

// Holds the random kernel + bias tensors in an allocated context so they persist
// across the separate single-shot / streaming compute contexts (mirrors how the
// loader previously handed out long-lived tensors).
struct ConvWeights {
    struct ggml_context*  ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    struct ggml_tensor*   w   = nullptr;   // kernel: ne=[K, C_out, C_in]
    struct ggml_tensor*   b   = nullptr;   // bias:   ne=[C_out]
    ~ConvWeights() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
};

// ggml_conv_transpose_1d(ctx, a=kernel, b=data): kernel ne=[K, C_out, C_in, 1],
// input ne=[T, C_in, 1], output ne=[T_out, C_out, 1] (see ggml.c). Fill with a
// deterministic normal distribution; vary the seed per case index.
//
// The weights are allocated on the active backend (not a plain no_alloc=false
// CPU context) so tensor->buffer is non-NULL: maybe_add_bias_t reshapes the
// bias into a view, and ggml_backend_alloc_ctx_tensors asserts that a view's
// view_src->buffer is set.
bool make_weights(ConvWeights* out, int K, int C_out, int C_in, unsigned seed) {
    struct ggml_init_params p{}; p.mem_size = ggml_tensor_overhead()*8; p.no_alloc = true;
    out->ctx = ggml_init(p);
    out->w = ggml_new_tensor_3d(out->ctx, GGML_TYPE_F32, K, C_out, C_in);
    out->b = ggml_new_tensor_1d(out->ctx, GGML_TYPE_F32, C_out);
    out->buf = vv::allocate_ctx_tensors(out->ctx);
    if (!out->buf) return false;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<float> wd(ggml_nelements(out->w)); for (auto& v : wd) v = nd(rng);
    std::vector<float> bd(ggml_nelements(out->b)); for (auto& v : bd) v = nd(rng);
    ggml_backend_tensor_set(out->w, wd.data(), 0, sizeof(float)*wd.size());
    ggml_backend_tensor_set(out->b, bd.data(), 0, sizeof(float)*bd.size());
    return true;
}

int run_case(const std::string& name, int K, int stride, int C_in, int C_out,
             int T, unsigned seed, double tol) {
    ConvWeights cw;
    if (!make_weights(&cw, K, C_out, C_in, seed)) { std::fprintf(stderr,"FAIL %s: weight alloc\n",name.c_str()); return 2; }

    // Input [T, C_in] contiguous (T fastest per channel), random.
    std::vector<float> in((size_t)T*C_in);
    { std::mt19937 rng(seed ^ 0x9e3779b9u); std::normal_distribution<float> nd(0.f,1.f);
      for (auto& v : in) v = nd(rng); }

    std::vector<float> ref; int refT=0, refC=0;
    if (!run_single(cw.w,cw.b,in,T,C_in,stride,&ref,&refT,&refC)) { std::fprintf(stderr,"FAIL %s: single-shot compute\n",name.c_str()); return 3; }
    if (refC != C_out) { std::fprintf(stderr,"FAIL %s: C_out mismatch got=%d want=%d\n",name.c_str(),refC,C_out); return 3; }

    // Chunk the input into 3 pieces along the time (frame) axis. Each chunk
    // emits a [emit_len, C_out] block; splice those blocks back into a single
    // [T_out, C_out] buffer (matching the single-shot ggml layout, ne0 = T_out
    // fastest) at the running time offset so the compare below is
    // apples-to-apples for C_out > 1.
    std::vector<float> got((size_t)refT*refC, 0.f);
    vv::StreamingCache cache; cache.is_first_chunk=true;
    int gotT=0, gotC=refC;
    const int nchunks=3; int done=0;
    for (int c=0;c<nchunks;++c) {
        const int t0=done, t1=(c==nchunks-1)? T : std::min(T,(int)((double)(c+1)*T/nchunks));
        if (t1<=t0) continue;
        cache.is_final_chunk=(t1==T);
        // seg is [seg_T, C_in] contiguous: input is [T, C_in] (T fastest).
        std::vector<float> seg((size_t)(t1-t0)*C_in);
        for (int ch=0; ch<C_in; ++ch)
            std::memcpy(seg.data()+(size_t)ch*(t1-t0), in.data()+(size_t)ch*T+t0, sizeof(float)*(t1-t0));
        std::vector<float> chunk_out; int sT=0,sC=0;
        if (!run_stream_chunk(cw.w,cw.b,seg,t1-t0,C_in,stride,cache,&chunk_out,&sT,&sC)) { std::fprintf(stderr,"FAIL %s: stream chunk compute\n",name.c_str()); return 4; }
        // Splice [sT, sC] block into got at time offset gotT, per channel.
        if (gotT + sT <= refT && sC == refC) {
            for (int ch=0; ch<sC; ++ch)
                std::memcpy(got.data()+(size_t)ch*refT+gotT, chunk_out.data()+(size_t)ch*sT, sizeof(float)*sT);
        }
        gotC=sC; gotT+=sT; done=t1;
    }
    if (gotT != refT) { std::fprintf(stderr,"FAIL %s: T mismatch stream=%d single=%d\n",name.c_str(),gotT,refT); return 5; }
    if (gotC != refC) { std::fprintf(stderr,"FAIL %s: C mismatch stream=%d single=%d\n",name.c_str(),gotC,refC); return 5; }
    double maxabs=0; for (size_t i=0;i<ref.size();++i) maxabs=std::max(maxabs,(double)std::fabs(ref[i]-got[i]));
    std::printf("%-24s max_abs=%.3e K=%d s=%d C_in=%d C_out=%d T=%d (transpose streaming)\n",
                name.c_str(), maxabs, K, stride, C_in, C_out, T);
    return maxabs < tol ? 0 : 6;
}

}  // namespace

int main() {
    const double tol = 1e-4;
    int rc=0, s;
    // Real-decoder ratios, kernel convention K = 2*stride. C_in=8 -> C_out=4.
    s=run_case("convt_k4_s2",  /*K=*/4,  /*stride=*/2, /*C_in=*/8, /*C_out=*/4, /*T=*/37, /*seed=*/1, tol); if (s) rc=rc?rc:s;
    s=run_case("convt_k10_s5", /*K=*/10, /*stride=*/5, /*C_in=*/8, /*C_out=*/4, /*T=*/41, /*seed=*/2, tol); if (s) rc=rc?rc:s;
    s=run_case("convt_k16_s8", /*K=*/16, /*stride=*/8, /*C_in=*/8, /*C_out=*/4, /*T=*/53, /*seed=*/3, tol); if (s) rc=rc?rc:s;
    return rc;
}
