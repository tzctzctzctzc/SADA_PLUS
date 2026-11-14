# 最新选择性边界版本归档

这次提交包含当前可复现的多 VLM A + 选择性边界证据 B 版本。

- 远端训练 root：`/vepfs/vlms/module_b_selective_boundary_formal6_1114_fix5`
- 代码 SHA256：`5ba5a7d60ce5a052c5d3167b4861ba2aab4aa35524ae6a082146263ebd3b6762`
- seed：123，S1–S6 各一次
- A：Qwen3-VL-8B、InternVL3.5-8B、VideoChat3-4B 的冻结 threshold router，阈值 0.8，不使用 top-K
- B：`path8`、`shared_joint_path`、VLM gate、selective boundary evidence
- 目标 TRAIN GT：未使用

`results/formal_status_snapshot.json` 是远端只读状态快照；只有在远端生成完整 `summary/result/audit` 后，才能把某个数字作为最终六向均值。当前主表文件明确标为 interim，不把冻结迁移行冒充完整外部检测器重训结果。
