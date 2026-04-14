# 07. CPU GPU Heterogeneous Scheduling | CPU 与 GPU 异构调度 (CPU & GPU Heterogeneous Scheduling)

**难度：** Hard | **标签：** `系统架构`, `异构计算`, `Offload` | **目标人群：** 核心 Infra 与算子开发

在深度学习系统中，GPU 并非孤立运行的，它必须接受 CPU 的指令和数据投喂。当模型的参数和中间状态庞大到连 GPU 集群的显存（如多张 80GB A100）都难以完全装下时，如何利用廉价但海量的 CPU 内存（Host RAM）参与训练和推理，成为了系统优化的关键。

本节将从 CPU 与 GPU 的基础交互开始，逐步深入到流水线重叠与极致的 Offload（卸载）技术。

> **相关阅读**:  
> 请前往实战篇进行相关代码练习：  
> [`../03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer.md`](../03_CUDA_and_Triton_Kernels/15_PyTorch_CUDA_Streams_and_Transfer.md)  

---

## Q1：在异构计算中，Host 和 Device 分别扮演什么角色？它们之间的核心物理瓶颈是什么？

<details>
<summary>点击展开查看解析</summary>

在典型的异构计算（如基于 NVIDIA GPU 的服务器）体系中，存在两个核心的实体：
1. **Host (宿主机)**：指代 CPU 及其直接挂载的系统内存（RAM）。它负责运行操作系统、网络通信、数据预处理以及向 GPU 下发计算指令，扮演“指挥官”的角色。
2. **Device (设备)**：指代 GPU 及其自带的显存（HBM）。它拥有极高的并发计算能力，负责执行密集的张量矩阵运算，扮演“执行者”的角色。

**核心物理瓶颈：PCIe 总线带宽**
Host 与 Device 之间的数据搬运必须通过 PCIe (Peripheral Component Interconnect Express) 总线。
- **带宽悬殊**：以 PCIe 4.0 x16 为例，其双向理论带宽仅约为 64 GB/s（单向 32 GB/s）。相比之下，A100 内部的 HBM 带宽高达 1.5 - 2.0 TB/s。
- **系统影响**：这意味着，如果频繁地在 CPU 内存和 GPU 显存之间拷贝数据（如反复执行 `tensor.to('cuda')` 和 `tensor.to('cpu')`），极慢的 PCIe 传输将使得 GPU 的计算核心长时间处于饥饿等待状态，导致严重的性能下降。
</details>

---

## Q2：既然跨设备传输 (PCIe) 如此缓慢，底层框架是如何利用 CUDA Streams 隐藏通信延迟的？

<details>
<summary>点击展开查看解析</summary>

为了解决 PCIe 带宽瓶颈，大模型训练框架（如 Megatron-LM、DeepSpeed）广泛采用了 **计算与通信重叠 (Overlap Computation and Communication)** 技术。

**核心机制：CUDA Streams 与异步执行**
在 CUDA 编程模型中，Stream（流）是一个按照顺序执行的指令队列。不同的 Stream 之间可以并行执行。通过 DMA (Direct Memory Access) 控制器，GPU 可以在不占用其计算核心（SM）的情况下，从 CPU 内存中异步拉取数据。

**重叠流水线的实现**：
假设我们需要处理一大批数据，可以将其切分为多个微块 (Micro-batches)：
- **阶段 1**：Stream 1 将数据块 A 从 Host 拷贝到 Device。
- **阶段 2**：Stream 1 开始在 GPU 上计算块 A。**同时**，Stream 2 开始将数据块 B 从 Host 拷贝到 Device。
- **阶段 3**：当块 A 计算完毕时，块 B 的数据刚好到达显存，Stream 2 无缝衔接开始计算块 B。

通过这种精密的异步调度，缓慢的 PCIe 传输时间被完全“隐藏”在了 GPU 密集的矩阵乘法计算时间之中，从而实现了逼近理论极限的硬件利用率。
</details>

---

## Q3：在资源极度受限的情况下，什么是 CPU Offload (卸载) 技术？它在训练和推理中分别如何应用？

<details>
<summary>点击展开查看解析</summary>

当我们在 06 节提到的 ZeRO-3 切分策略依然无法将模型塞进 GPU 显存时，系统就会采用 **Offload (卸载) 技术**：将部分显存压力转移到廉价且容量庞大的 CPU 内存（甚至 NVMe 固态硬盘）上。这本质上是用 PCIe 带宽换取显存容量。

1. **训练期的 ZeRO-Offload**：
   - 优化器状态（如 Adam 的动量和方差）通常占据了最多的显存。ZeRO-Offload 策略会将这部分状态全部转移到 CPU 内存中保存。
   - **执行流程**：GPU 完成反向传播算出梯度后，将梯度通过 PCIe 传给 CPU。CPU 利用自身的计算能力执行参数更新（Optimizer Step），然后将更新后的权重再通过 PCIe 传回 GPU，准备下一轮前向传播。

2. **推理期的 KV Cache Offload (如 vLLM 中的实现)**：
   - 在处理超长上下文或面临极高的并发请求时，GPU 显存可能无法容纳所有用户的 KV Cache。
   - **执行流程**：推理引擎会将当前暂时不活跃请求的 KV Cache “踢出”显存，保存到 CPU 内存中（换出，Evict）。当该请求再次被调度执行时，再从 CPU 内存拉回显存（换入，Swap-in）。这一机制完美复刻了操作系统中的**虚拟内存分页与缺页中断机制**，极大提升了单机的并发承载上限。
</details>
