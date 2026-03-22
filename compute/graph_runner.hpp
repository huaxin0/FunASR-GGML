// funasr/compute/graph_runner.hpp
// GGML 计算图执行封装
//
// 旧代码中 build_graph → plan → compute 这套流程重复了 10+ 次。
// 封装成一个函数，自动管理 work_data 内存。
//
#ifndef FUNASR_COMPUTE_GRAPH_RUNNER_HPP
#define FUNASR_COMPUTE_GRAPH_RUNNER_HPP

#include <ggml.h>
#include <ggml-cpu.h>
#include <vector>
#include <cstdio>

namespace funasr {

// 构建并执行计算图
// ctx: 包含所有 tensor 的 ggml_context
// output: 计算图的输出 tensor
// n_threads: CPU 线程数
// max_nodes: 计算图最大节点数（70层 encoder 需要较大值）
inline bool run_graph(
    struct ggml_context* ctx,
    struct ggml_tensor* output,
    int n_threads = 4,
    int max_nodes = 131072
) {
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(gf, output);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);

    std::vector<uint8_t> work_buf;
    if (plan.work_size > 0) {
        work_buf.resize(plan.work_size);
        plan.work_data = work_buf.data();
    }

    ggml_graph_compute(gf, &plan);
    return true;
}

} // namespace funasr

#endif // FUNASR_COMPUTE_GRAPH_RUNNER_HPP