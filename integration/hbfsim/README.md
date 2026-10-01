# Qwen2.5-1.5B P32D2：LLMCompass 与 HBFSim 联合模拟

## 1. 范围与严格术语

本目录实现算子边界闭环联合模拟（operator-boundary analytical closed-loop co-simulation）：LLMCompass 计算每个高层算子的分析型计算时长，HBFSim 执行该算子的内存事务有向无环图（Directed Acyclic Graph，DAG）；HBFSim 返回的完成时间成为下一算子的实际提交时间。因此，内存完成时间会反馈并推进后续计算，而不是把固定 trace 事后单向重放。

这里的“闭环”只到高层算子边界。它不表示线程块（Cooperative Thread Array，CTA）、warp、指令、缓存、流水线或 GPU 周期级联合模拟，也不表示真实硬件校准或硬件性能预测。

- Qwen2.5-1.5B：28 个 Transformer 层，隐藏维度 1536，门控前馈中间维度 8960，12 个查询头、2 个键值头，每头 128 维；权重采用 Brain Floating Point 16（BF16，16 位浮点）字节数。
- P32D2：批大小 1，先执行 32 token 的 prefill（提示预填充），再执行 2 个逐 token decode（自回归解码）步骤；三个阶段的上下文长度分别为 32、33、34。
- Grouped-Query Attention（GQA，分组查询注意力）：12 个查询头共享 2 个键值头。
- Root Mean Square Layer Normalization（RMSNorm，均方根归一化）、Rotary Position Embedding（RoPE，旋转位置编码）和 SwiGLU（Swish-Gated Linear Unit，门控前馈结构）均作为独立算子族建模。
- Key/Value cache（KV cache，键值缓存）：prefill 写入 32 个位置，两个 decode 分别追加第 33、34 个位置。

## 2. 两个模拟器的职责

主脚本复用 LLMCompass 现有 Matmul、BatchedMatmul、Softmax 的 roofline 接口及调用者显式选定的设备配置；默认是 `configs/GA100.json`，面向 RTX 4000 Ada 的运行必须改用本目录的 `RTX4000Ada_xmu_candidate_v0.json`。向量算子用相同设备组件的向量吞吐和片上带宽构造分析时长。BF16 在当前 LLMCompass 数据类型表中没有原生条目，因此使用同为 2 字节的 FP16 吞吐代理；这是模型假设，不是 BF16 硬件校准。

每个算子的 HBFSim 事务关系为：输入、KV cache 和权重读取；依赖全部读取的计算 barrier（屏障），其 duration_ns 取自 LLMCompass；依赖屏障的输出或 KV cache 写入；本批次 HBFSim 完成 frontier（完成前沿时间）成为下一算子的提交时间。

权重放在只读 HBF_STATIC 区域，激活和 KV cache 放在 HBM 区域。地址是适配器确定性分配的模拟逻辑地址，不是 CUDA 虚拟地址或真实模型进程地址。默认配置每个 plane 的一个 static block 覆盖 1 GiB 逻辑空间；适配器按权重 footprint 向上取整设置 static_blocks_per_plane。

## 3. 目录边界与来源

最终实现文件、构建目录和最终运行输出都位于 LLMCompass/integration/hbfsim。适配器只读取、不修改 LLMCompass/configs/GA100.json、HBFSim 客户端与配置以及当前 HBFSim C++ 源码。调试期默认 wear 输出造成的目录边界偏差在第 6 节单独列出。

- qwen25_1p5b.json：模型和 P32D2 自包含描述。
- qwen_hbfsim_cosim.py：计划生成、事务 DAG 和持久 HBFSim 会话。
- test_qwen_hbfsim_cosim.py：模型、权重、算子族、地址和依赖验证。
- _build/hbfsim-current/：从当前 HBFSim 源码编译的二进制（Git 忽略）。
- results/<run-id>/：不可复用的单次运行目录（Git 忽略）。

## 4. 构建与验证

从当前 HBFSim 源码构建到本 integration 目录，以免误用旧二进制：

    cd /home/xmu/nvidiagds/simulators/LLMCompass
    cmake -S /home/xmu/nvidiagds/simulators/HBFSim -B integration/hbfsim/_build/hbfsim-current -DCMAKE_BUILD_TYPE=Release
    cmake --build integration/hbfsim/_build/hbfsim-current --target hbfsim -j2
    integration/hbfsim/_build/hbfsim-current/hbfsim --version

验证、最小闭环和完整 P32D2：

    python3 -m unittest integration/hbfsim/test_qwen_hbfsim_cosim.py
    python3 integration/hbfsim/qwen_hbfsim_cosim.py plan
    python3 integration/hbfsim/qwen_hbfsim_cosim.py smoke --output integration/hbfsim/results/<fresh-smoke-run-id>
    python3 integration/hbfsim/qwen_hbfsim_cosim.py run --output integration/hbfsim/results/<fresh-full-run-id>

运行目录必须事先不存在，避免覆盖证据。manifest.json 先记录 RUNNING，终态写成 PASS 或 FAILED；失败目录原样保留。

## 5. RTX 4000 Ada 候选配置与近似对比

RTX4000Ada_xmu_candidate_v0.json 是单卡 RTX 4000 Ada 的候选 LLMCompass 配置。它只使用 XMU GPU0 实测的 Streaming Multiprocessor（SM，流式多处理器）数量、计算能力、显存容量、共享内存、驱动、最大时钟与功耗上限；频率、L2 带宽、Tensor Core 效率、Graphics Double Data Rate 6（GDDR6，第六代图形双倍数据率内存）控制器拓扑仍是明确记录的近似假设。不得把该文件称为微架构校准配置。

使用该候选配置生成对比模型结果：

    python3 integration/hbfsim/qwen_hbfsim_cosim.py run \
      --architecture integration/hbfsim/RTX4000Ada_xmu_candidate_v0.json \
      --output integration/hbfsim/results/<fresh-ada-candidate-run-id>

硬件端必须使用同一份 Qwen P32D2 合同、同一模型权重 SHA、BF16、TP1、FlashInfer、禁用 Compute Unified Device Architecture（CUDA，统一计算设备架构）Graph 与 torch.compile，并使用无 NVIDIA Binary Instrumentation（NVBit，NVIDIA 二进制插桩框架）插桩的计时。候选模拟与硬件的 phase 时间可计算相对误差，但这个误差只能命名为候选配置 phase 误差（Candidate Profile Phase Error，候选 LLMCompass 配置在单阶段时间上的偏差），不能命名为 Ada 硬件准确度。HBFSim 的 HBM/HBF 结果在 RTX 4000 Ada 对比中是 GDDR6 bus surrogate（GDDR6 总线路径替代模型）/反事实内存路径，不是物理硬件匹配项。

`compare` 子命令读取不可变的硬件 `finish.json` 和模拟 `summary.json`，拒绝非 PASS 或非固定 P32D2 合同，并将结果写入一个新的目录：

    python3 integration/hbfsim/qwen_hbfsim_cosim.py compare \
      --hardware-finish integration/hbfsim/results/<hardware-run-id>/finish.json \
      --simulation-summary integration/hbfsim/results/<ada-candidate-run-id>/summary.json \
      --output integration/hbfsim/results/<fresh-comparison-run-id>

输出状态 `CANDIDATE_PROFILE_COMPARISON_ONLY` 的严格含义是“只比较候选配置与指定硬件收据”，不表示校准通过。`hardware_host_wall_ms` 是 SGLang 驱动在同步执行周围取得的一次宿主机墙钟时间，包含运行时开销；`simulated_operator_boundary_ms` 是 LLMCompass 分析算子计划加 HBFSim completion frontier（完成前沿）的阶段耗时；`signed_error_percent=(simulated_operator_boundary_ms-hardware_host_wall_ms)/hardware_host_wall_ms*100`。因此这个数不是 CUDA event 内核时间误差、数值误差、缓存误差或统计置信区间。

## 6. 归档 NCU 流量兼容性筛查

`compare-ncu-traffic` 读取用户指定的归档 `comparison.json`、候选联合模拟 `traffic.json` 与 `manifest.json`。它固定选择 `qwen1p5b-P32D2`、`r4-small-shared`、`phase=whole`、`scope=whole` 的同一硬件窗口，拒绝缺少任一 L1、L2 或 DRAM 行的输入，且从不把 Prefill、Decode、steps 与 whole 混为同一个分母。示例中的 `P32D2` 是 Prompt length 32（提示长度 32）与 Decode steps 2（解码步数 2）；它仅表示形状，不单独证明模型权重、SGLang 运行时或 kernel 序列相同。

    python3 integration/hbfsim/qwen_hbfsim_cosim.py compare-ncu-traffic \
      --archived-ncu-comparison /absolute/path/to/traffic-comparison/comparison.json \
      --simulation-traffic integration/hbfsim/results/<ada-candidate-run-id>/traffic.json \
      --simulation-manifest integration/hbfsim/results/<ada-candidate-run-id>/manifest.json \
      --hardware-finish integration/hbfsim/results/<hardware-run-id>/finish.json \
      --output integration/hbfsim/results/<fresh-ncu-compatibility-run-id>

输出 `ncu-compatibility.json` 是自包含收据。NVIDIA Nsight Compute（NCU，NVIDIA 内核性能分析器）的 `dram__bytes_*` 和 cache sector 行保留三次硬件采样的最小值、中位数和最大值；Level 1 cache（L1，一级缓存）和 Level 2 cache（L2，二级缓存）在当前适配器中始终为 `NOT_MODELED`，所以绝不产生 L1/L2 误差。对 Dynamic Random Access Memory（DRAM，动态随机存取内存），报告只可把 High Bandwidth Flash（HBF，HBFSim 闪存层）静态权重目标 `HBF_STATIC` 与 High Bandwidth Memory（HBM，高带宽内存）目标的 `physical_bytes` 相加，作为同形状下的量级筛查代理；这两个目标均不是 RTX 4000 Ada 的 Graphics Double Data Rate 6（GDDR6，第六代图形双倍数据率内存）域。

因而 `signed_screening_delta_percent=(HBF_STATIC+HBM physical_bytes-NCU DRAM hardware_median)/NCU DRAM hardware_median*100` 只表示筛查方向，不能叫“DRAM 误差”或“硬件准确度”。如果归档没有 whole NCU range 的匹配持续时间，硬件带宽显示 `NOT_AVAILABLE`，模拟侧仅保留串行算子边界的区间平均 GB/s，二者不比较。

若同时传入固定 P32D2 的 `--hardware-finish`，收据会额外写入 `cross_receipt_host_time_normalized_proxy`。它把归档 NCU whole-window 的 DRAM 字节中位数除以独立 SGLang host-synchronized（宿主机同步）P32D2 时间，再与模拟 completion bytes 除以模拟算子边界时间并列。该字段的状态恒为 `CROSS_RECEIPT_HOST_TIME_NORMALIZED_PROXY_ONLY`：它补充一个可复现的速率诊断，不能替代同一 NCU ROI（Region of Interest，感兴趣区间）的 GPU duration，也不能叫硬件 DRAM 带宽或硬件准确度。归档缺少 SGLang app/issue 合同、权重 SHA-256（Secure Hash Algorithm 256-bit，256 位安全散列）或运行时 identity 时，`contract_join` 必须是 `PARTIAL_LABEL_MATCH_ONLY`；最终校准结论始终是 `NOT_CALIBRATED`。

## 7. 输出和声明边界

- operators.jsonl：每行一个算子，记录 phase、层号、计算屏障、读取完成时间、算子完成时间和 completion 数量。
- summary.json：终态、模拟总完成时间、各阶段完成时间、算子数和声明边界。
- traffic.json：逐阶段和总计的 HBFSim completion receipt（完成收据）流量；其中 `logical_bytes` 是适配器请求字节数，`physical_bytes` 是按 HBFSim 传输粒度完成的字节数，`effective_*_GBps` 是字节数除以该串行算子边界阶段的 ns，GB/s 为每秒十亿字节。这是模型区间平均带宽，不是物理链路峰值或硬件实测。
- manifest.json：两个仓库的 Git 提交、HBFSim 二进制与配置 SHA-256（Secure Hash Algorithm 256-bit，256 位安全散列）、计划摘要及输出散列。

结果是该分析模型和当前 HBFSim 配置下的算子边界模拟时间。除非以后补齐真实 trace、微架构耦合和独立硬件校准，否则不得解释为 GPU 实测延迟、周期精度或硬件加速比。

- 只模拟 batch size 1、P32D2 和 Tensor Parallelism 1（TP1，单设备张量并行度 1），不模拟跨设备通信。
- 权重缓存驻留未建模；每个投影算子显式提交权重读取。
- 因果性在算子边界验证：写完成时间不得早于全部读完成时间加计算屏障时间。
- 没有采样 token、停止条件或数值张量内容；“full inference”严格指完整 P32D2 算子序列，不代表文本生成到自然终止。

## 8. 当前验证状态（2026-09-30）

最终 smoke 证据目录是 results/qwen-p32d2-smoke-r7：1 个 prefill 层、21 个算子、65 条内存 completion，终态 PASS，模拟终点 724,937.5 ns。

最终完整证据目录是 results/qwen-p32d2-full-r5：完整 28 层 prefill 加两个 28 层 decode，共 1521 个算子和 4812 条内存 completion，终态 PASS。累计阶段 frontier 为 prefill 4,831,454.5 ns、decode_1 9,607,662.5 ns、decode_2 14,383,834.5 ns。独立离线审计验证了逐算子索引、相邻 batch frontier 连续性、读完成加计算屏障不晚于写完成、阶段分母、结果散列及 static HBF 容量；3,221,225,472 字节 static 容量覆盖 3,087,716,352 字节权重 footprint。

最终 manifest 绑定适配器、模型描述、LLMCompass 架构配置、HBFSim 配置及 HBFSim 二进制的 SHA-256，并记录引擎启动握手。当前 HBFSim 引擎提交为 d7a2ca64614a6d9ce8d7a69beb77ce78b66df1a8；握手如实报告 git_dirty=true，因此本结果不能被描述成 clean-tree 证据。

此前调试证据不作为最终结果：smoke-r1 是客户端导入前留下的空目录；smoke-r2 因记录字段名错误失败；smoke-r3 虽为 PASS 但错误地重复累加绝对 frontier；smoke-r4 修正相对时间后 PASS，但还未发现 FFN scratch 重叠；smoke-r5 已修正 scratch，但 manifest 未包含最终增强的来源字段；smoke-r6 使用增强 manifest，但 wear 报告仍落到根目录。full-r1 因同一绝对时间问题触发 HBM 时钟上限并 FAILED；full-r2 虽 PASS 但存在 FFN scratch 重叠；full-r3 已修正 scratch 并 PASS，但使用增强前 manifest；full-r4 使用增强 manifest，但 wear 报告仍落到根目录。以上目录按失败证据保留，未删除、覆盖或重标。

HBFSim 在不传 hbf_wear_output_prefix 时会默认向当前工作目录的 out/hbf-wear 写报告。调试运行因此在 LLMCompass/out 下生成 8 个 run 子目录、16 个报告文件；这违反本任务的 integration-only 目标。最终适配器已经把前缀强制指向各运行的 results 目录，smoke-r7 和 full-r5 的 wear HTML/JSON 均位于各自证据目录且散列进入 manifest，之后没有继续写根目录 out。由于没有当前删除授权，旧 out 诊断文件尚未删除或移动。

RTX 4000 Ada 候选对比的硬件收据是 `results/qwen-p32d2-hardware-gpu0-baseline-r2/finish.json`。它在 GPU0（Universally Unique Identifier，UUID，全局唯一标识符为 `GPU-18ace299-5348-e6e4-d48c-1ee5a602859b`，48 SM、compute capability 8.9、20,900,347,904 bytes 显存）上以无 NVBit 插桩的 SGLang baseline PASS；测得一份 phase 样本：Prefill 12.353697791695595 ms、Decode1 10.915396735072136 ms、Decode2 11.003632098436356 ms，总和 34.272726625204086 ms。该收据的 `numerical_acceptance` 是 `NOT_ASSESSED`，所以仅证明该固定形状执行完成，不证明 logits 或生成文本一致。r1 在 FlashInfer just-in-time（JIT，即时编译）初始化阶段因 `/usr/local/cuda/bin/nvcc` 不存在而未生成任何 timing 收据；该空目录原样保留。r2 仅通过进程环境将 JIT 编译器定向到现有 `/usr/local/cuda-12.8/bin/nvcc`，没有修改系统 CUDA 链接、驱动或 GPU 配置。

对应的候选联合模拟收据是 `results/qwen-p32d2-ada-candidate-r1/summary.json`：1521 个算子、4812 条 memory completion、PASS；累计阶段 frontier 为 8,478,893.5 ns、16,796,038.5 ns、25,113,154.0 ns，换算阶段耗时为 8.4788935 ms、8.317145 ms、8.3171155 ms，总和 25.113154 ms。`results/qwen-p32d2-ada-comparison-r1/comparison.json` 将两份收据的 SHA-256 固定在一起，状态是 `CANDIDATE_PROFILE_COMPARISON_ONLY`；三阶段 signed error 依次为 -31.365542180417567%、-23.803548310101483%、-24.41481662057855%，总和为 -26.7255439736976%。这说明当前候选模型在这一单样本合同下低估 9.159572625204085 ms；其校准结论严格为 `NOT_CALIBRATED`，不是“已与 RTX 4000 Ada 一致”。

候选联合模拟 r2 (`results/qwen-p32d2-ada-candidate-r2/traffic.json`) 新增可审计流量收据，仍为 1521 个算子和 4812 条 completion。总计 HBF_STATIC 权重目标为 1,017 个读事务、9,262,390,272 logical bytes、9,263,255,552 physical bytes，按 25,113,154 ns 的完整模型区间折算为 368.826244/368.860700 GB/s logical/physical；HBM 目标为 2,106 个读事务、1,689 个写事务，物理读/写分别为 115,065,088/99,585,536 bytes，对应 4.581865/3.965473 GB/s。HBF_STATIC 是适配器把权重映射到 HBF 的反事实目标，不是 RTX 4000 Ada 的显存；HBM 是 HBFSim 的 High Bandwidth Memory（HBM，高带宽内存）目标，也不是该卡的 GDDR6。

Level 1 cache（L1，一级缓存）和 Level 2 cache（L2，二级缓存）在当前适配器中的状态均为 `NOT_MODELED`：没有 CTA/warp/指令级访问或缓存替换策略，候选配置中的 L2 带宽仅参与 roofline 时间计算，不产生 L2 hit/miss 或字节计数。Dynamic Random Access Memory（DRAM，动态随机存取内存）行只能报告上述 HBFSim HBM/HBF 目标流量，不能称为 Ada DRAM 流量。

真实硬件计数器 gate 收据是 `results/qwen-p32d2-hardware-gpu0-ncu-counter-gate-r1/status.json`。它以 `/usr/local/cuda-12.8/bin/ncu --query-metrics --devices 0` 对 GPU0 进行无 workload 的计数器可用性检查，返回 `ERR_NVGPUCTRPERM`（当前用户无 NVIDIA GPU performance counter 权限）。这意味着本任务不能新采集硬件计数器，也不得用 root/驱动修改绕过 gate。已有归档 NCU P32D2 比较材料可以作为只读外部硬件分母，但必须通过本节的 `compare-ncu-traffic` 生成 `SCREENING_PROXY_ONLY_NOT_HARDWARE_ACCURACY` 收据：L1/L2 仍为 `NOT_MODELED`，HBF/HBM 与 GDDR6 的域差异仍在，且归档缺少完整 runtime/weight 合同；故不构成硬件准确度验收或带宽对比。

2026-09-30 已用归档 `traffic-comparison/comparison.json`（SHA-256 `2a509c6a8f76e06d3c67a4e36bfea2875f9bf03ab91bd994931b4a75050fbab1`）和 r2 `traffic.json`/`manifest.json` 生成 `results/qwen-p32d2-archived-ncu-compatibility-r1/ncu-compatibility.json`（SHA-256 `662d46951f5824e6f2cc87e878e77dd99ef70dd9cddb862cecf3ade4242f038a`）。该收据固定 P32D2/r4/whole：硬件 DRAM 读中位数 9,240,374,144 bytes、聚合 HBF_STATIC+HBM physical read 9,378,320,640 bytes，筛查差值 +1.4928669970530577%；硬件 DRAM 写中位数 72,884,480 bytes、聚合 physical write 99,585,536 bytes，筛查差值 +36.63476229781703%。六个硬件 L1/L2 计数全部为 `NOT_COMPARABLE_SIM_NOT_MODELED`；归档 whole 窗口没有匹配持续时间，故硬件 bandwidth（带宽）为 `NOT_AVAILABLE`，不与 373.44256480090075/3.9654730743896205 GB/s 的模拟串行区间均值进行数值比较。该结论仍是 `NOT_CALIBRATED`。

## 9. GDDR 数值覆盖层的全 HBM 测试

HBFSim 目录中的 `configs/ramulator2/GDDR6_RTX3070_SM86.yaml` 是 Ramulator2（Ramulator2，独立 DRAM 时序模拟器）的 YAML 输入，不是当前 Python `SimulationSession` 接受的 `key=value` 系统配置；不得把该 YAML 直接交给本适配器并声称已接入原生 GDDR6 后端。当前会话实际调用的是 HBFSim 的 generic HBM device（泛型 High Bandwidth Memory，HBM，高带宽内存设备）。

`gddr_abstract_all_hbm` 是一个受限的、可运行的数值覆盖模式。它把现有 `configs/overlays/gddr6/rtx4000-ada-uncalibrated.cfg` 作为后置 `key=value` 覆盖层，保留其 20 GiB、10 个 x16 channel、32-byte burst 和标记为未校准的 GDDR-inspired（GDDR 启发式）时序数值；底层实现仍是 HBM device，`hbm-standard` 仍为 JEDEC HBM4，因而绝不是 GDDR6 device identity。该模式禁用 High Bandwidth Flash（HBF，HBFSim 闪存层），将 HBF_STATIC 权重映射到 HBM 地址 `[0, weight_footprint)`，并把原有 KV cache（键值缓存）和 activation（激活）HBM 地址整体平移 `weight_footprint`，从而保证单一 20 GiB HBM 地址空间无重叠。没有添加 Level 1 cache（L1，一级缓存）、Level 2 cache（L2，二级缓存）、CTA、warp、指令或 32-byte sector 缓存模型。

默认命令保持 LLMCompass 的 `configs/GA100.json`，因此它只能用于“默认 LLMCompass 配置 + GDDR 数值覆盖”的跨配置诊断。面向 RTX 4000 Ada 的对比必须显式传入本目录的单卡候选配置；该文件与 XMU GPU0 以及 NVIDIA RTX 4000 Ada 公开板卡规格对齐 48 SM、20 GB GDDR6、160-bit、360 GB/s、130 W 和 2.175 GHz boost-equivalent 频率。候选文件中的 L2/片上带宽、Tensor Core 效率和控制器地址映射仍是未校准近似，不能叫 Ada accuracy（Ada 硬件准确度）。运行目录必须新建：

    python3 integration/hbfsim/qwen_hbfsim_cosim.py smoke \
      --architecture integration/hbfsim/RTX4000Ada_xmu_candidate_v0.json \
      --memory-layout gddr_abstract_all_hbm \
      --hbfsim-overlay /home/xmu/nvidiagds/simulators/HBFSim/configs/overlays/gddr6/rtx4000-ada-uncalibrated.cfg \
      --output integration/hbfsim/results/<fresh-gddr-smoke-id>

    python3 integration/hbfsim/qwen_hbfsim_cosim.py run \
      --architecture integration/hbfsim/RTX4000Ada_xmu_candidate_v0.json \
      --memory-layout gddr_abstract_all_hbm \
      --hbfsim-overlay /home/xmu/nvidiagds/simulators/HBFSim/configs/overlays/gddr6/rtx4000-ada-uncalibrated.cfg \
      --output integration/hbfsim/results/<fresh-gddr-full-id>

    python3 integration/hbfsim/qwen_hbfsim_cosim.py compare \
      --hardware-finish integration/hbfsim/results/<hardware-run-id>/finish.json \
      --simulation-summary integration/hbfsim/results/<fresh-gddr-full-id>/summary.json \
      --output integration/hbfsim/results/<fresh-gddr-diagnostic-id>

`traffic.json.phase_coverage` 是已完成的连续 P32D2（Prompt length 32，提示长度 32；Decode steps 2，解码步数 2）阶段前缀；smoke 只有 `prefill`，不能作为 full P32D2 分母。`physical_bytes` 仍是 HBFSim completion receipt（完成收据）的传输粒度字节数；`effective_physical_GBps` 是串行算子边界的区间平均，既不是实际 RTX 4000 Ada GDDR6 带宽，也不是 NCU（NVIDIA Nsight Compute，NVIDIA 内核性能分析器）范围带宽。

2026-09-30 的默认配置结果如下。`qwen-p32d2-gddr-abstract-default-smoke-r1` 在 traffic receipt 强制三阶段的旧路径上 FAILED；该失败收据按证据要求保留。修正为显式 `phase_coverage=["prefill"]` 后，`qwen-p32d2-gddr-abstract-default-smoke-r2` PASS：21 个算子、65 个 completion、1.7352168888888888 ms，HBM physical read/write 为 564,346,880/3,810,048 bytes，HBF_STATIC read 为 0。完整 `qwen-p32d2-gddr-abstract-default-full-r1` PASS：1,521 个算子、4,812 个 completion、29.04918355555555 ms；三阶段 HBM physical read/write 分别为 prefill 3,194,058,752/93,168,384 bytes、decode_1 3,091,683,072/3,207,680 bytes、decode_2 3,091,713,536/3,209,472 bytes。总 HBM physical read/write 为 9,377,455,360/99,585,536 bytes，串行区间均值为 322.8130436804168/3.4281698764285795 GB/s。

`qwen-p32d2-gddr-abstract-default-hardware-diagnostic-r1/comparison.json` 使用既有 RTX 4000 Ada 无插桩 SGLang P32D2 收据，只产生 `GDDR_ABSTRACT_CROSS_CONFIG_DIAGNOSTIC_ONLY`：硬件总时间 34.272726625204086 ms、模拟总时间 29.049183555555555 ms、signed diagnostic delta 为 -15.241107387724295%。阶段差值为 prefill -18.48473988364646%、decode_1 -13.063250549377308%、decode_2 -13.759897907673219%。这由默认 A100×4 LLMCompass 配置、GDDR 数值覆盖与真实 RTX 4000 Ada 的差异共同造成，不是性能准确度。`qwen-p32d2-gddr-abstract-default-ncu-compatibility-r1/ncu-compatibility.json` 的 whole NCU 流量筛查同样不升级结论：DRAM read 为 +1.483502874058516%，DRAM write 为 +36.63476229781703%，L1/L2 为 `NOT_MODELED`，合同 join 为 `PARTIAL_LABEL_MATCH_ONLY`。

2026-09-30 的 RTX 4000 Ada 基础配置对齐结果是当前推荐诊断收据。`qwen-p32d2-rtx4000ada-gddr-abstract-aligned-smoke-r1` PASS：21 个算子、65 条 completion、2.381078222222222 ms。完整 `qwen-p32d2-rtx4000ada-gddr-abstract-aligned-full-r1` PASS：1,521 个算子、4,812 条 completion；prefill、decode_1、decode_2 的算子边界时间分别为 13.719310222222221、13.033321777777777、13.03354844444444 ms，总计 39.78618044444444 ms。总 HBM physical read/write 为 9,377,455,360/99,585,536 bytes，HBF_STATIC 为零，说明权重、KV cache 和 activation 都在单一 GDDR-inspired HBM 代理地址空间内。

`qwen-p32d2-rtx4000ada-gddr-abstract-aligned-hardware-diagnostic-r1/comparison.json` 将该全流程与既有 GPU0 SGLang 收据绑定：硬件总时间 34.272726625204086 ms，模拟总时间 39.78618044444444 ms，signed diagnostic delta 为 +16.087000837528265%；三个阶段分别为 +11.054280698404478%、+19.40309724062123%、+18.447693705576913%。`qwen-p32d2-rtx4000ada-gddr-abstract-aligned-ncu-compatibility-r1/ncu-compatibility.json` 的流量筛查为 DRAM read +1.483502874058516%、DRAM write +36.63476229781703%，但 L1/L2 仍是 `NOT_MODELED`，归档硬件 whole 窗口没有匹配持续时间，故硬件带宽仍为 `NOT_AVAILABLE`。这三项百分比均是跨模型边界诊断，不是硬件准确度或 GDDR6 后端验证。

## 10. 验收关键节点：Prefill 与单步 Decode 流量（2026-10-01）

本节记录当前阶段的验收基线。Prefill 指 32-token 提示预填充；Decode 1 和 Decode 2 分别指第一个和第二个逐 token 解码步骤。`DRAM` 表示真实 RTX 4000 Ada 的 NVIDIA Nsight Compute（NCU）计数器；`HBM proxy` 表示 LLMCompass/HBFSim 在 generic HBM device 上、使用 GDDR-inspired 数值覆盖层得到的 `physical_bytes`，不是物理 GDDR6 流量。所有字节使用十进制 GB/MB，带宽使用十进制 GB/s。

硬件列来自 `qwen-p32d2-solid-bandwidth-r1/summary.json` 的 5 次计数器中位数和 20 次未插桩自然 CUDA-event timing；模型列来自 `qwen-p32d2-rtx4000ada-gddr-abstract-aligned-full-r1/traffic.json`。模型相对差异定义为 `(模型-硬件)/硬件`，只作方向性筛查，不作硬件准确度结论。

| 阶段 | 硬件 DRAM read | 模型 HBM proxy read | read 差异 | 硬件 DRAM write | 模型 HBM proxy write | write 差异 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 3.095 GB | 3.194 GB | +3.21% | 66.746 MB | 93.168 MB | +39.59% |
| Decode 1 | 3.060 GB | 3.092 GB | +1.03% | 3.066 MB | 3.207 MB | +4.60% |
| Decode 2 | 3.088 GB | 3.092 GB | +0.12% | 0.902 MB | 3.209 MB | +255.76% |

对应的阶段带宽（read/write/total）为：

| 阶段 | 硬件 DRAM GB/s | 模型 HBM proxy GB/s |
|---|---:|---:|
| Prefill | 255.265 / 5.505 / 260.770 | 232.815 / 6.791 / 239.606 |
| Decode 1 | 280.746 / 0.281 / 281.027 | 237.214 / 0.246 / 237.460 |
| Decode 2 | 283.290 / 0.083 / 283.372 | 237.212 / 0.246 / 237.458 |

验收判断：Prefill 的读流量差异为 +3.21%，Decode 1/2 的读流量差异收敛到 +1.03%/+0.12%；但 Prefill 写流量仍高估 +39.59%，Decode 2 的写流量绝对值很小而相对差异为 +255.76%。模型的 Decode 读带宽低于硬件约 15--16%，说明当前算子边界模型的时间/服务路径仍未校准。L1/L2 在当前 LLMCompass 适配器中均为 `NOT_MODELED`，本验收节点不产生 L1/L2 误差或命中率结论。

本节点状态：`PASS_DIRECTIONAL_TRAFFIC_SCREENING_NOT_HARDWARE_ACCURACY`。原始摘要、identity 和 SHA-256 收据保留在上述结果目录；未启动新实验，未修改或删除历史结果。
