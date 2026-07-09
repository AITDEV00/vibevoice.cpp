// Streaming SConvTranspose1d parity: chunked (with StreamingCache) must equal
// single-shot, bit-exact modulo fp noise. Uses in-repo fixtures (no model).
#include "conv1d.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model_loader.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool file_ok(const std::string& p) { FILE* f = std::fopen(p.c_str(), "rb"); if (!f) return false; std::fclose(f); return true; }

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

int run_case(const std::string& path, double tol) {
    if (!file_ok(path)) { std::fprintf(stderr, "skip: missing %s\n", path.c_str()); return 77; }
    vv::ModelLoader loader; if (!loader.load(path)) return 1;
    const int stride = loader.get_i32("convt.stride");
    const int T = loader.get_i32("convt.T");
    struct ggml_tensor* in_t = loader.tensor("test.input");
    struct ggml_tensor* w = loader.tensor("weight.kernel");
    struct ggml_tensor* b = loader.tensor("weight.bias");
    if (!in_t || !w || !b) return 2;
    const int C_in = (int)in_t->ne[1];
    std::vector<float> in((size_t)T*C_in); std::memcpy(in.data(), in_t->data, sizeof(float)*in.size());

    std::vector<float> ref; int refT=0, refC=0;
    if (!run_single(w,b,in,T,C_in,stride,&ref,&refT,&refC)) return 3;

    // Chunk the input into 3 pieces along the time (frame) axis. Each chunk
    // emits a [emit_len, C_out] block; we splice those blocks back into a
    // single [T_out, C_out] buffer (matching the single-shot ggml layout, ne0
    // = T_out fastest) at the running time offset so the flat compare below is
    // apples-to-apples for C_out > 1.
    std::vector<float> got((size_t)refT*refC, 0.f);
    vv::StreamingCache cache; cache.is_first_chunk=true;
    int gotT=0, gotC=refC;
    const int nchunks=3; int done=0;
    for (int c=0;c<nchunks;++c) {
        const int t0=done, t1=(c==nchunks-1)? T : std::min(T,(int)((double)(c+1)*T/nchunks));
        if (t1<=t0) continue;
        cache.is_final_chunk=(t1==T);
        // seg is [seg_T, C_in] contiguous: input is [T, C_in] row-major (T fastest).
        std::vector<float> seg((size_t)(t1-t0)*C_in);
        for (int ch=0; ch<C_in; ++ch)
            std::memcpy(seg.data()+(size_t)ch*(t1-t0), in.data()+(size_t)ch*T+t0, sizeof(float)*(t1-t0));
        std::vector<float> chunk_out; int sT=0,sC=0;
        if (!run_stream_chunk(w,b,seg,t1-t0,C_in,stride,cache,&chunk_out,&sT,&sC)) return 4;
        // Splice [sT, sC] block into got at time offset gotT, per channel.
        if (gotT + sT <= refT && sC == refC) {
            for (int ch=0; ch<sC; ++ch)
                std::memcpy(got.data()+(size_t)ch*refT+gotT, chunk_out.data()+(size_t)ch*sT, sizeof(float)*sT);
        }
        gotC=sC; gotT+=sT; done=t1;
    }
    if (gotT != refT) { std::fprintf(stderr,"FAIL %s: T mismatch stream=%d single=%d\n",path.c_str(),gotT,refT); return 5; }
    if (gotC != refC) { std::fprintf(stderr,"FAIL %s: C mismatch stream=%d single=%d\n",path.c_str(),gotC,refC); return 5; }
    double maxabs=0; for (size_t i=0;i<ref.size();++i) maxabs=std::max(maxabs,(double)std::fabs(ref[i]-got[i]));
    std::printf("%-24s max_abs=%.3e T=%d s=%d C=%d (transpose streaming)\n", path.c_str(), maxabs, T, stride, C_in);
    return maxabs < tol ? 0 : 6;
}
}  // namespace

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string fix = VV_FIXTURES_DIR;
    int rc=0, s;
    s=run_case(fix+"/sconvt1d_basic.gguf", 2e-3); if (s!=0 && s!=77) rc=rc?rc:s;
    s=run_case(fix+"/sconvt1d_long.gguf",  2e-3); if (s!=0 && s!=77) rc=rc?rc:s;
    return rc;
}
