# Ascend GEMM 独立测试

119个单元按四个二级特性组织，覆盖清单中的14个三级特性。每个 `test_<编号>.py` 只有一个测试单元、一个固定场景，合法与非法条件分别测试。

每个文件均有完整kernel定义，数值文件同时有独立CPU参考；内核从原有用例复制或按固定场景改写。重复内核也保留在各自文件，不调用其他测试文件或公共kernel工厂。你列出的FP4两种L1表示、显式L0、SF绑定、有效M/N、有效K24及MX尾部补零均有独立文件，原用例到文件编号的对应关系见执行指南。

- [逐用例执行指南](EXECUTION_GUIDE.md)：每个单元的三级特性、场景、约束责任、输入输出、执行文件、命令和验收标准。
- [本轮执行报告](TEST_REPORT.md)：每个单元独立一条实际结果，区分 IR、设备编译、仿真和真机。
- [main直接执行版报告](DIRECT_MAIN_TEST_REPORT.md)：本次按示例改写后的实际编译、跳过和约束验证结果。
- [原IR单元测试命令](IR_TEST_COMMANDS.md)：38个原有IR参数场景，以及新增K96负例的准确pytest节点命令。

所有完整DSL用例的 `main()` 都直接调用 `tilelang.compile`。53个数值用例在自己的文件中调用 `make_inputs()`、`ref_program()`，然后执行 `result = kernel(*inputs)`；需要初值或保留GM哨兵时使用 `out_idx=None`，传入 `initial.clone()` 并原地更新结果。最后与CPU参考比较，输出 `max_diff` 和 `correctness passed / 运行结果正确`。pytest与一键入口均调用这个main，不再通过公共硬件执行器运行。

第一部分打印 `program.script()`（输入TIR）、`kernel.get_kernel_source()`（实际编译的AscendC源码）及编译后的设备IR。第二部分输出验收状态。IR局部用例在对应pass边界检查，不作为已初始化的完整数值内核启动。

在仓库根目录一键运行真机验收并生成报告：

```bash
python testing/ascend/gemm_features/run.py --report gemm-report.md
```

独立运行、失败时按需保存材料：

```bash
python testing/ascend/gemm_features/scaled_gemm/test_s03.py --failure-dir gemm-failures
```

单独运行L1打包FP4的两种输入表示（需要支持对应指令的昇腾NPU）：

```bash
python testing/ascend/gemm_features/scaled_gemm/test_s10.py
python testing/ascend/gemm_features/scaled_gemm/test_s24.py
```

种子写死在每个文件中。默认只输出到终端；指定 `--failure-dir` 且用例 FAIL 时才写日志和复现材料。未完成数值验收不会报告“运行结果正确”。

编译器负责的约束未拦截判为 FAIL。调用方负责的有效范围约束由 T08 单列为 OBSERVED，记录当前是否提供静态防护，不启动非法指令，也不将观察结果当作数值 PASS。
