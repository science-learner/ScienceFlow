# CLI/TUI 本地验证任务

激活含 pytest、click 和 Textual 的开发环境后，在本仓库 checkout 中运行：

```bash
python tests/support/validate_cli_tui.py
```

若解释器或 InquiryCraft 不在默认位置，可分别传入 `--python` 和 `--inquirycraft`。

任务首先确认该解释器实际导入的是这两个本地源码目录，避免误测已安装的发行包。随后运行 InquiryCraft 的恢复会话折叠、模型代理路由测试，以及 ScienceFlow 的 CLI/TUI、Web 启停和代理配置回归；再检查 `--help`、`--version`、`tui --help`、`agent tools list`。这些调用不启动研究任务，也不访问模型端点。

最后对 `--help` 和 `--version` 各预热一次、计时五次，取中位数。默认本机验收线分别为 **3.5 秒**和 **6.5 秒**，约为此前 7 秒、13 秒观测值的一半；共用机器上可用 `--max-help-seconds`、`--max-version-seconds` 调整。任一功能检查或耗时门槛失败，任务以非零状态退出。计时反映当前机器与缓存条件，不代替跨机器性能基准。

需要目视验收时，在已有会话的工作区用同一源码组合运行 `scienceflow tui --resume --workspace <工作区>`：最近两轮应可见，其余历史和欢迎提示应收进 History；模型配置警告若出现，仍须单独可见。这一步需要交互终端，不计入自动任务。

此任务验证本地源码组合，不能充当发行安装验证；发行后需另用干净环境重复安装与 TUI 恢复测试。
