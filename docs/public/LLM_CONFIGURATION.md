# 模型服务配置

ScienceFlow 使用一个结构化模型注册表，不再要求维护多组模型环境变量：

```bash
scienceflow config init
scienceflow config path
```

默认文件是 `~/.config/scienceflow/models.json`，创建权限为 `600`，已有文件不会被覆盖。

```json
{
  "version": 1,
  "models": {
    "deepseek-v4-flash": {
      "model": "deepseek-v4-flash",
      "use_proxy": false,
      "reasoning_replay": "preserve",
      "pricing": {
        "input_usd_per_1m": 0.15,
        "cached_input_usd_per_1m": 0.003,
        "output_usd_per_1m": 0.60
      },
      "endpoints": {
        "code": [
          {"url": "https://provider-a.example.com/v1", "key": "key-a"}
        ],
        "feedback": [
          {"url": "https://provider-b.example.com/v1", "key": "key-b"}
        ]
      }
    },
    "qwen3-coder": {
      "model": "qwen3-coder",
      "reasoning_replay": "preserve",
      "endpoints": [
        {"url": "https://provider-c.example.com/v1", "key": "key-c"}
      ]
    }
  },
  "defaults": {
    "code_models": ["deepseek-v4-flash", "qwen3-coder"],
    "feedback_models": ["deepseek-v4-flash"],
    "selection": "auto"
  }
}
```

`models` 的键使用干净模型名，不添加 `code-` 或 `feedback-`。同一个模型可以为
Code 和 Feedback 配置不同 endpoint；旧的共享 endpoint 数组仍兼容。请求限流、
超时或连接失败时，InquiryCraft 在对应角色的 endpoint 内执行冷却和故障转移。

`use_proxy` 可在每个模型 alias 下设为 `true` 或 `false`，同时作用于该 alias
的 Code、Feedback 端点；省略时默认 `false`，即直连。设为 `true` 时，
ScienceFlow 使用环境中的 `HTTPS_PROXY` / `HTTP_PROXY`（也接受小写形式，可配
`USER_NAME` / `USER_PWD`）；TUI 模型切换沿用同一路由设置，`NO_PROXY`
不会覆盖显式启用的代理。`false` 明确直连。代理地址与密码不要写入 `models.json`。
若代理替换 HTTPS 证书，需要通过系统信任链或 `SSL_CERT_FILE` 配置其 CA。

`pricing` 是聊天、Long Research、TUI 和 Web 费用估算的唯一价格配置，三个字段均为
USD/百万 token。三个值必须同时填写；缺少价格的模型显示 `Cost —`。价格按实际请求的
`model` 精确匹配，不从环境变量、聊天 profile 或内置默认表推断。

`reasoning_replay` 控制历史 assistant reasoning 如何进入下一轮模型请求：`preserve`
（默认）保留已有真实 reasoning，`required` 为缺失 reasoning 的旧消息补稳定兼容占位符，
`omit` 不向 provider 发送 reasoning。三种策略都不会删除或改写 canonical memory 和
JSONL 记录。Stage 显式配置优先于模型 alias；同一模型池的 alias 策略必须一致，否则
启动时会报错，避免 endpoint 切换改变上下文前缀。

TUI、普通 agent 和 Long Research 默认使用同一个文件：

```bash
scienceflow tui
scienceflow agent run "检查工作区"
```

普通 TUI chat 一次固定使用一个模型 alias；未显式指定 `--model` 时选择
`defaults.code_models` 的第一个 alias。同一 alias 下的多个 endpoint 仍可故障转移。
这项限制不影响 Long Research 显式配置多个 Code 模型并分配给不同 worker。

在 TUI 输入 `/models` 后，底部输入框会原位切换为单列聊天模型选择卡。可以用上下方向键移动、
Enter 确认、数字 1–9 快速选择或 Esc 返回输入框。也可以输入 `/models <alias>` 或
`/models <index>` 直接切换。该命令只影响当前 TUI 对话，不修改 Feedback 或
Long Research 配置，也不会清空对话。选择卡底部会显示当前配置文件路径；编辑后重新打开
`/models` 即可读取新配置。如果配置缺失或无效，TUI 会直接提示 `scienceflow config init`
和 `scienceflow config path`。

临时选择模型或配置文件：

```bash
scienceflow agent run "检查工作区" --models deepseek-v4-flash,qwen3-coder
scienceflow agent run "检查工作区" --model-config /path/models.json
```

Long Research 可以在任务级覆盖默认模型：

```text
/long-research circle packing workers=4 duration=1h models=auto
/long-research circle packing code-models=deepseek-v4-flash feedback-models=deepseek-v4-flash model-policy=fixed
```

如果命令中没有提供模型配置，TUI 会在运行资源配置完成后、preflight 前显示一次
模型问询。输入 `defaults` 可接受 `defaults.code_models` 中的第一个 Code 模型和默认
Feedback 模型，也可以输入 `models=... feedback-models=... model-policy=...` 覆盖本次
任务。启动命令已经显式提供模型配置时不会重复问询。同名 Code 模型的多个 endpoint
仍可故障转移；只有显式提供多个 `models=` 时，`auto` 才会把不同模型分配给不同
worker。这些选择不影响普通聊天。

`auto` 使用默认模型池并为不同 worker 自动选择主模型，同时保留故障转移；
`fixed` 让所有 worker 固定使用所列的第一个模型。TUI 只展示这两个选项。
旧配置中的 `spread` 仍兼容，其现有运行行为不变。一个 stage 内保持 sticky，
以保留上下文和缓存。生成的 manifest 只保存模型别名，不复制 URL 或 key。

任务、数据、CPU、GPU、worker 数和时间通过 `/long-research` 交互提供，无需写入模型配置。
