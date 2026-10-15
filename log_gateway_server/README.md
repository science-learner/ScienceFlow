# log_gateway_server

基于 FileBeat 机制的日志中间服务：实时监控多个日志文件，增量读取并把每行/每段落日志转成 **SSE（Server-Sent Events）** 消息推送给下游前端。

## 特性

- **多文件监控**：`inputs[].path` 支持 glob 通配符（如 `./logs/*.log`），按 `scan.frequency` 周期扫描新文件。
- **三种读取模式**：`raw`（完全同步源文件原始内容，base64 分块）、`prefix`（按 strftime 时间戳行前缀聚段）、`line`（普通逐行）。
- **增量读取 + 检查点续读**：仿 FileBeat registry，把每个文件的 offset 持久化到 `data/registry.json`，重启后从上次位置继续，不漏不重。
- **多行合并**：`line` 模式用仿 FileBeat `multiline`；`prefix` 模式用 strftime 前缀自动聚段。
- **轮转/截断检测**：支持 `rename`（新文件替换）与 `copytruncate`（原地截断）两种主流日志轮转方式。
- **SSE 广播**：`GET /events` 推流，支持 `?source=` 按输入名或路径过滤，慢消费者丢弃、不阻塞采集。
- **主动拉取快照**：`GET /read_cached_content?path=…` 或 `?source=…` 一次性返回源文件全部已有内容。
- **健康检查 / 演示页**：`GET /healthz`、`GET /`（浏览器实时 tail 页面）。
- **主动调用 ScienceFlow agent**：开启 `agent.enabled` 后，`POST /sessions/{id}/invoke` 可在 `$SCIFLOW_WORKSPACE_ROOT/<user>/<session>/` 下按 `doc/agent_launch.md` 的方式启动 agent 子进程执行用户消息。agent 的 stdout/stderr 实时写入 `$SCIFLOW_WORKSPACE_ROOT/task_logs/<user>/<session>/RAW.log`，由 tailer 监听并通过 SSE 流式推送至前端。

## 目录结构

```
log_gateway_server/
├── go.mod
├── main.go                     # 入口：装配 config/registry/hub/tailer/server
├── config.json                 # 示例配置
├── internal/
│   ├── config/     # 配置加载与默认值
│   ├── hub/        # SSE 事件广播
│   ├── server/     # HTTP + SSE 接口
│   └── tailer/     # registry / multiline / harvester / tailer
└── logs/           # 示例日志目录
```

## 快速开始

```powershell
# 1. 编译
go build -o scienceflow_lgw.exe .

# 2. 运行
.\scienceflow_lgw.exe -config config.json
```

另开一个终端推流：

```powershell
curl -N http://127.0.0.1:8080/events
```

再开一个终端追加日志，即可在 SSE 流中实时看到：

```powershell
Add-Content -Path logs\sample.log -Value "20260828-10:00:03.456789 INFO new line appearing"
```

浏览器打开 `http://127.0.0.1:8080/` 可实时查看。

## 读取模式

每个 `inputs[]` 通过 `mode` 选择读取方式：

| mode | 行为 | 去向字段 |
|------|------|----------|
| `line`（默认） | 按 `\n` 逐行，可选 `multiline` 合并 | `message` |
| `prefix` | 按 `prefix_format`（strftime 时间戳）识别新条目起始行，非匹配行作为延续行聚段 | `message` |
| `raw` | 不做任何解析，文件新增字节原样分块推送（`base64`） | `data` + `encoding` |

- `raw` 模式"完全同步源文件内容"：不关心行前缀、换行、编码，每读到一个块（≤32KB）就作为一条 SSE 消息 `base64` 输出，客户端解码后按原始字节还原。
- `prefix` 模式的 `prefix_format` 遵循 strftime 语法（`%Y %m %d %H %M %S %f …`），内部转换为锚定 `^` 的正则做条目起始判定；`match` 语义等同 `after`。
- 三种模式都具备增量续读与轮转/截断检测。

## 启动与部署

### 本地开发

```powershell
go run . -config config.json
```

### 编译为单一可执行文件（推荐，Go 静态/单二进制，无运行时依赖）

```powershell
# 当前平台
go build -o scienceflow_lgw.exe .

# 交叉编译（示例：目标 Linux amd64）
$env:GOOS="linux"; $env:GOARCH="amd64"; go build -o scienceflow_lgw .; Remove-Item Env:GOOS,Env:GOARCH
```

### 前台运行

```powershell
.\scienceflow_lgw.exe -config config.json
```

### 启动引导（scienceflow 环境自检）

`scienceflow_lgw` 启动时默认先执行一次引导流程：检测 `agent.python` 解释器（显示解析后的真实路径与版本）→ 检测 scienceflow 安装与版本 → 查询索引最新版本（scienceflow 只发预发布版，自动回退 `--pre`）→ 按需安装/更新 → 配置启动端口并检测占用 → 打印网关配置总览后进入服务流程。

- **未安装**：显示 Python 路径与 pip 安装命令，询问是否自动安装；确认则执行 `pip install`，拒绝则打印手动命令并以退出码 1 结束。
- **有更新**：显示新旧版本并询问是否自动更新；确认则执行 `pip install --upgrade`，拒绝则保留当前版本继续启动。
- **启动端口**：交互式运行可自定义端口（默认 `8901`，回车采用默认）；非交互/`-yes` 运行使用 `config.json` 的 `server.port`。
- **端口被占用**：显示占用进程的名称、PID 与完整命令，询问是否中止（默认否）。确认后先发 `SIGTERM`（Windows 用 `taskkill`），超时未退出再强制终止，并等待端口真正释放后才继续；拒绝则打印 `kill <pid>` 等手动命令并以退出码 1 结束。非交互/`-yes` 模式**永不**自动中止进程，遇占用直接报错退出。
- `agent.enabled=false` 时跳过安装/更新引导，仅做检测与配置展示。

```text
◆ [4/4] 配置启动端口
  ? 启动端口 [8901]
  ⚠ 端口 8901 已被以下进程占用
  进程          scienceflow_lgw（PID 7147）
  命令          ./scienceflow_lgw -config config.json
  ? 是否中止该进程以释放端口 8901？ [y/N]
  ✓ 端口 8901 已释放
```

端口中止的安全性：端口号与 PID 均做数值校验（PID 仅来自系统工具解析，绝不拼接进 shell）；拒绝 PID<2 及自身 PID；发送信号前会再次核对占用者 PID，防止 PID 复用误杀；只对占用端口的那个进程发信号。

命令开关：

| 参数 | 默认 | 说明 |
|------|------|------|
| `-bootstrap` | `true` | 是否执行启动引导；后台/服务化部署可设 `-bootstrap=false` |
| `-yes` | `false` | 所有确认自动回答"是"（无人值守自动安装/更新，隐含开启 bootstrap） |

非交互环境（stdin 为 `/dev/null`、nohup/systemd）不会挂起等待输入：缺失 scienceflow 时打印手动安装命令并以退出码 1 结束；有可用更新时跳过更新、保留当前版本继续启动；端口被占用时打印手动中止命令并以退出码 1 结束（不会自动杀进程）。设置 `NO_COLOR=1` 可关闭彩色输出。

### 后台 / 守护运行

```powershell
# 方式一：Windows 后台进程（日志重定向到文件）
Start-Process -FilePath .\scienceflow_lgw.exe `
  -ArgumentList "-config", "config.json" `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "scienceflow_lgw.out.log" -RedirectStandardError "scienceflow_lgw.err.log"

# 方式二：注册为 Windows 服务（用 NSSM / WinSW 包裹 scienceflow_lgw.exe，
#         工作目录设为项目根目录，参数设 -config config.json）

# 方式三：Linux 上交给 systemd/supervisor 托管，或用 nohup 后台运行
#   nohup ./scienceflow_lgw -config config.json > scienceflow_lgw.out.log 2>&1 &
```

> 后台运行建议加 `-bootstrap=false`（跳过交互引导）或 `-yes`（无人值守自动安装/更新）。

### 停止

```powershell
Stop-Process -Name scienceflow_lgw
```

> 收到 Ctrl+C / SIGTERM 时会优雅退出：关闭 HTTP、停掉所有 harvester、把 registry 检查点落盘，重启后从上次 offset 无缝续读。

### 生产部署要点

1. `registry.path` 建议放独立的持久化目录（如容器 volume），避免重建丢 offset。
2. `inputs[].path` 建议用绝对路径并指向日志卷。
3. 前端访问建议经反向代理（Nginx/Caddy）加 CORS 与鉴权；SSE 需关闭代理缓冲（`X-Accel-Buffering: no` 已内置）。
4. 服务为无状态（除 registry 外的状态都在客户端/日志源），可多实例水平扩展，但多个实例读同一批日志会造成重复推送，通常单实例即可。

## 同步时延调优

端到端时延 ≈ `poll_interval`（EOF 后轮询，主导项）+ 读取 + SSE 推送（发布即 Flush，可忽略）。

- 默认 `poll_interval = 250ms`；要控制在 **50ms 以内**，设 `"poll_interval": "30ms"`（示例配置已如此），最坏时延约 30ms。
- **注意：** 若开启了聚段（`line` 模式的 `multiline` 或 `prefix` 模式），每条日志会被缓冲，直到下一条匹配行或超时才吐出——这才是更大的时延来源。`prefix` 模式用 `prefix_timeout`（默认 50ms）；`line` 模式的 `multiline.timeout` 默认 5s，纯单行场景请**不配 multiline**。
- 代价：轮询会让每个 harvester 周期性唤醒，CPU 占用与「文件数量 × 轮询频率」成正比。几十个文件用 30ms 无压力；上百个文件时建议回到 100~250ms，或用 fsnotify 事件驱动（见下）。
- `scan.frequency` 只管新文件发现，不影响已监控文件的实时性。
- 如需亚毫秒级，可改用文件系统通知（Go 侧 `fsnotify`，Windows 底层 `ReadDirectoryChangesW`）替代轮询，本服务预留了 poll 架构，替换 harvester 的 EOF 等待逻辑即可。

多行日志示例（Java 堆栈 / JSON 跨行，`match: after`）：

```json
{
  "name": "stack-logs",
  "path": "./logs/*.log",
  "tail_files": true,
  "poll_interval": "30ms",
  "multiline": {
    "pattern": "^\\d{4}-\\d{2}-\\d{2}T",
    "negate": false,
    "match": "after",
    "timeout": "50ms"
  }
}
```

## 配置说明（config.json）

| 字段 | 含义 | 默认 |
|------|------|------|
| `server.host` / `server.port` | 监听地址 | `0.0.0.0:8080` |
| `server.sse_path` | SSE 端点 | `/events` |
| `auth.enabled` | 是否开启用户鉴权 | `false` |
| `auth.file` | 用户账密文件（YAML） | `AUTH.yaml` |
| `auth.token_secret` | HMAC 签名密钥，留空则每次启动随机（重启后 token 失效） | 空 |
| `auth.token_ttl` | 登录 token 有效期 | `24h` |
| `auth.session_ttl` | 会话空闲过期时间 | `1h` |
| `auth.reload_interval` | AUTH.yaml 热更新轮询间隔（修改文件后自动生效） | `5s` |
| `inputs[].path` | 日志 glob 模式 | 必填 |
| `inputs[].name` | 输入逻辑名（SSE 里 `input` 字段，过滤用） | 实际路径 |
| `inputs[].mode` | 读取模式：`line` / `prefix` / `raw` | `line` |
| `inputs[].prefix_format` | strftime 行前缀（`mode=prefix` 必填，如 `%Y%m%d-%H:%M:%S.%f`） | 空 |
| `inputs[].prefix_timeout` | prefix 模式挂起条目的超时刷新 | `50ms` |
| `inputs[].tail_files` | `true` 只读新增内容（从文件尾开始）；`false` 首次读到历史全文 | `false` |
| `inputs[].ignore_older` | 忽略超过该时长的文件 | `0s`（不忽略） |
| `inputs[].poll_interval` | EOF 后的轮询间隔（实时性关键，越小越及时） | `250ms`（示例 `30ms`） |
| `inputs[].multiline.pattern` | 新事件起始行的正则 | 空（关闭） |
| `inputs[].multiline.negate` | 反转 pattern 含义（匹配行视为延续行） | `false` |
| `inputs[].multiline.match` | `after` / `before` | `after` |
| `inputs[].multiline.timeout` | 挂起多行的超时刷新 | `5s` |
| `registry.path` | 检查点文件 | `data/registry.json` |
| `registry.flush_interval` | 检查点落盘间隔 | `5s` |
| `scan.frequency` | 扫描新文件周期 | `10s` |

## 多用户多端鉴权与会话

开启 `auth.enabled` 后，采用「登录发 token → 建会话订阅 → SSE 推流」模型，支持每个用户按权限只读自己可见的日志源，同一用户可开多个会话（多端）。

**账密与权限文件**（`AUTH.yaml`）：

```yaml
# 简化写法：只有密码，可读所有日志源
sciflow: "123456"

# 完整写法：sources = 可读源（config.json 里 inputs[].name 列表，或 ["*"] 全量）
#           role: admin = 超级管理员，可通过 /users 接口管理账号
viewer:
  password: viewer456
  sources: ["prefixed-logs", "plain-text"]

admin:
  password: admin123
  role: admin
```

**热更新**：AUTH.yaml 修改后无需重启，网关按 `auth.reload_interval`（默认 5s）轮询文件变化并自动生效（无效 YAML 会被拒绝并保留原账号，日志输出原因）。通过 `/users` 接口的增删改会原子写回 AUTH.yaml（临时文件 + rename，权限 `0600`，会丢失人工注释并写入统一文件头）。密码修改或账号删除会**立即失效该账号已签发的 token**（token 与密码派生标记绑定）；token 签名密钥不受热更新影响，其他用户不受牵连。

### 控制面接口（Bearer token 鉴权）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/login` | POST | 传 `{"user_name","password"}`，返回 `token`、`role` 与该用户可见的 `sources` |
| `/sessions` | POST | 传 `Authorization: Bearer <token>` + `{"sources":[...]}`，创建会话返回 `session_id` |
| `/sessions/{id}` | GET | 查询会话语订阅信息 |
| `/sessions/{id}/sources` | PUT | 替换会话订阅源（自动按权限过滤） |
| `/sessions/{id}` | DELETE | 关闭会话 |
| `/sources` | GET | 列出当前用户可见的源及各自文件路径 |
| `/users` | GET | **[超级管理员]** 列出全部账号（不含密码） |
| `/users` | POST | **[超级管理员]** 新建账号：`{"name","password","sources"?,"role"?}` |
| `/users/{name}` | PUT | **[超级管理员]** 修改账号：`password` / `sources` / `role` 任意组合 |
| `/users/{name}` | DELETE | **[超级管理员]** 删除账号 |

用户管理接口要求当前 token 对应账号在 AUTH.yaml 中带 `role: admin`（普通用户调用返回 `403`）。用户名限 `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`（会用作工作区目录名）；密码非空且不含换行/空字符。为防止锁死，**最后一个管理员不可删除、不可降级**（返回 `409`）；重复创建返回 `409`，目标不存在返回 `404`。

```bash
# 超级管理员通过接口管理账号（热更新，无需重启）
TOK=$(curl -s -X POST http://127.0.0.1:52001/login -H 'Content-Type: application/json' \
  -d '{"user_name":"admin","password":"admin123"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# 新建
curl -s -X POST http://127.0.0.1:52001/users -H "Authorization: Bearer $TOK" \
  -H 'Content-Type: application/json' \
  -d '{"name":"alice","password":"pw","sources":["prefixed-logs"]}'

# 改密码 / 改权限 / 改角色
curl -s -X PUT http://127.0.0.1:52001/users/alice -H "Authorization: Bearer $TOK" \
  -H 'Content-Type: application/json' -d '{"password":"newpw"}'

# 删除
curl -s -X DELETE http://127.0.0.1:52001/users/alice -H "Authorization: Bearer $TOK"
```

> 安全提示：AUTH.yaml 明文存储密码（与旧版一致），请限制文件权限（接口写回的版本为 `0600`）并避免提交到版本库；生产环境请务必修改示例口令。

SSE 推流用会话连接：`GET /events?session=<session_id>`；会话订阅变更会实时生效（无需重连）。

```powershell
# 1. 登录拿 token
$tok = (Invoke-RestMethod -Uri http://127.0.0.1:8080/login -Method Post `
  -ContentType application/json -Body '{"user_name":"viewer","password":"viewer456"}').token

# 2. 建会话（只订阅可见源里的 prefixed-logs）
$sid = (Invoke-RestMethod -Uri http://127.0.0.1:8080/sessions -Method Post `
  -ContentType application/json -Headers @{Authorization="Bearer $tok"} `
  -Body '{"sources":["prefixed-logs"]}').session_id

# 3. 推流
curl -N "http://127.0.0.1:8080/events?session=$sid"
```

鉴权失败（用户不存在/密码错误/token 无效/会话无效/越权访问源）统一返回 `401`，响应体固定：

```
用户不存在或密码错误
```

演示页 `http://127.0.0.1:8080/` 内置登录→建会话→推流的 JS 逻辑。

### 重启后的会话恢复

会话本身保存在内存中，但网关**启动时会扫描任务工作区** `$SCIFLOW_WORKSPACE_ROOT/<user>/<session>/`（`agent.enabled=true` 时），把仍留有任务产物的历史会话重新注册到会话列表。判定内容为以下任一文件存在：`gateway_first_user.txt`、`gateway_manifest*.yaml`、`task_logs/<user>/<session>/RAW.log`、`run/.logs/interaction.log`（纯空目录不恢复）：

- **会话 ID 保留**：前端本地保存的 `session_id` 在网关重启后依旧可用（`activate` / SSE 订阅 / invoke 均照常），无需重新建会话；
- **名称与时间**：会话名取自首问文件或 manifest 的 `task`（首行、折叠空白、截断 120 字符），列表 `last_active` 取自磁盘文件最新修改时间，排序与重启前一致；
- **历史回放**：SSE 连接后自动 backfill 历史 RAW.log / interaction.log / task_logs/parallel_subprocess.log（heavy）内容，前端据此重建对话；
- **续聊**：再次 invoke 复用同一工作区，ScienceFlow memory 自动加载，实现跨重启多轮对话；
- **TTL**：恢复的会话不受空闲 TTL 清理（内容在磁盘）；`DELETE /sessions/{id}` 仅从内存列表移除（软删除），目录仍在时下次重启会再次出现；
- 提示：`workspace_root` 变化或目录被清理后无法恢复；启动日志会打印 `restored N historical session(s) from task workspaces`。

### SSE 消息格式

`event: log`，`data` 为 JSON。普通/前缀模式：

```json
{
  "input": "prefixed-logs",
  "file": "sample.log",
  "path": "logs/sample.log",
  "message": "20260828-10:00:03.456789 INFO new line appearing",
  "mode": "prefix",
  "timestamp": "2026-08-28T10:00:03.123456789Z",
  "offset": 1204
}
```

`raw` 模式（内容在 `data`，`base64`）：

```json
{
  "input": "raw-logs",
  "file": "sample.raw",
  "path": "logs/sample.raw",
  "data": "UkFX...base64...",
  "encoding": "base64",
  "mode": "raw",
  "timestamp": "2026-08-28T10:00:03.123456789Z",
  "offset": 4096
}
```

未开启鉴权时，SSE 可用 `?source=a,b` 静态过滤（逗号分隔，匹配 `input` 或 `path`）；开启鉴权后由会话订阅决定。

## 主动调用 ScienceFlow agent

开启 `agent.enabled` 后，gateway 可作为 ScienceFlow agent 的前端驱动：用户在会话里发消息，gateway 在 `$SCIFLOW_WORKSPACE_ROOT/<user>/<session>/` 下生成 per-invocation manifest 并以子进程方式调用 `python -m scienceflow.cli repl --auto-first-user --exit-after-auto`，不改后端 agent 代码。agent 工作目录的日志（`task_logs/*.log`）自动纳入 `agent-logs` 源经 SSE 推送。

### 配置（config.json 的 `agent` 段）

| 字段 | 含义 | 默认 |
|------|------|------|
| `agent.enabled` | 是否开启 agent 调用 | `false` |
| `agent.python` | 解释器命令（支持 `uv run python`） | `python` |
| `agent.module` | CLI 模块 | `scienceflow.cli` |
| `agent.command` | 子命令：`repl` / `run` | `repl` |
| `agent.config_yaml` | 传给 CLI 的 `-c` 配置 yaml 路径；留空时使用安装包内置配置（site-packages 模式） | 空 |
| `agent.repo_root` | 子进程工作目录（ScienceFlow 源码仓根，源码包会遮蔽 site-packages）。**留空 = 安装包模式**：子进程 cwd 回退到任务工作区，`python -m scienceflow.cli` 从 site-packages 解析模块 | 空 |
| `agent.workspace_root` | 覆盖 `$SCIFLOW_WORKSPACE_ROOT`；为空时读环境变量 | 空 |
| `agent.input_data_dir` | 可选 `-d` 共享数据根 | 空 |
| `agent.exp_id` | 可选 `--exp-id` | 空 |
| `agent.timeout` | 单次调用 wall-clock 上限（含排队等待时间） | `3600s` |
| `agent.max_concurrent` | 同时运行的 agent 子进程上限 | `4` |
| `agent.max_queue` | 并发满后的等待队列上限；超出返回 `503` | `16` |
| `agent.log_dir_name` | workspace 下日志子目录名 | `task_logs` |
| `agent.model_store_path` | 模型 registry 文件路径；默认跟随安装版 CLI 的官方位置，`/models` 接口直接读写该文件 | `~/.config/scienceflow/models.json`（或 `$XDG_CONFIG_HOME/scienceflow/models.json`） |

### 模型配置（/models 接口 ↔ 官方 registry）

`/models` CRUD、`POST /models/{id}/activate` 与 `GET|PUT /models/stages` 直接管理安装版 scienceflow CLI 读取的同一份模型 registry（schema v1，权限 `0600`）：`model_name→alias/model`、`api_url/api_key→endpoints.code[0]`、激活模型→`defaults.code_models/feedback_models` 队首。`/models/stages` 的 per-session code/feedback 阶段模型选择在任务启动时物化为 per-task registry 副本（`<workspace_root>/task_logs/<user>/<session>/xdg/`，经 `XDG_CONFIG_HOME` 注入子进程），不污染共享 registry，API key 也不会进入 manifest/工作区（与上游"keys 不进 manifest/session/report"的约定一致）。

> 注意：旧版网关通过 `CODE_MODEL`/`FEEDBACK_*` 环境变量注入模型配置，安装版 CLI（0.2.0b5+）不消费这些变量，已移除。

### 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/sessions/{id}/invoke` | POST | `{"query":"..."}` 触发 agent；返回 task 快照（`202`）。同一 session 有任务在跑时返回 `409`；队列满返回 `503`。 |
| `/sessions/{id}/agent` | GET | 查询当前 task 状态 + 最近 stdout/stderr（≤256KB）+ 全队列统计。无任务时 `{"status":"idle"}`。 |
| `/sessions/{id}/agent` | DELETE | 杀掉当前 session 的 agent 子进程。 |

鉴权同其他控制面接口（`Authorization: Bearer <token>`），且只能操作属于当前用户的 session。

```bash
# 触发
curl -X POST "$BASE/sessions/$SID/invoke" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"分析 dataset/ 下的数据并产出 best_solution.json"}'

# 轮询状态与输出
curl "$BASE/sessions/$SID/agent" -H "Authorization: Bearer $TOKEN"

# 停止
curl -X DELETE "$BASE/sessions/$SID/agent" -H "Authorization: Bearer $TOKEN"
```

> 续聊：同一 `user/session` 的 workspace 重复 invoke 会自动加载已有 memory（scienceflow CLI 既有行为），实现多轮。

### RAW.log 实时流式输出

agent 子进程的 stdout/stderr（含 REPL 交互、LLM 回答、工具调用与结果）在运行过程中被实时追加写入：

```
$SCIFLOW_WORKSPACE_ROOT/task_logs/<user>/<session>/RAW.log
```

该文件被 gateway 的 tailer 自动监听（源名 `agent-logs`），新增的每一行通过 SSE `event: log` 推送至已订阅该源的会话。前端无需轮询，打开 SSE 连接即可实时看到 agent 的逐行输出。

**transcript 说明**：安装版 CLI（0.2.0b5+）在 `--plain` 模式下 stdout 几乎为空（仅 banner 与一行提示），完整交互（user/assistant/tool 轮次）写入 `<session>/run/.logs/interaction.log`；heavy（LNR）任务的完整研究过程由 `cli parallel` 的 worker 写入 `<session>/run/task_logs/parallel_subprocess.log`（RAW.log 此时只含 parallel 启动横幅与最终汇总）。任务启动时 gateway 同时 tail 这三个文件（存在哪个推哪个），均经 SSE `event: log` 推送，以 `file` 字段区分：

- `RAW.log`：stdout/stderr 镜像（banner、API 报错、traceback、任务 header、heavy 的启动/汇总）
- `interaction.log`：lite 任务的完整交互记录；推送前会剥离 ANSI 色彩码与 loguru 时间戳前缀，保留 `[user] ...`、`[tool-call] ...`、`[assistant] ...`、工具输出等有效载荷
- `parallel_subprocess.log`：heavy 任务的 LNR worker 全程输出（Agent Iteration、工具与阶段进度、最终 `Result: {...}` 统计）

SSE 事件与 backfill 均携带 `session` 字段做跨会话隔离；三个文件在 SSE 重连/会话切换/网关重启后都会自动回放（backfill）。注意：前端旧版聊天解析器（`agentLogParser.ts`）面向源码仓时代的 RAW.log 格式（`=== Task: ===`/`→ tool`/`## Summary`），对接安装版交互格式需相应适配。

SSE 消息示例（`file` 字段为 `RAW.log`）：

```json
{
  "input": "agent-logs",
  "file": "RAW.log",
  "path": "/.../task_logs/admin/<session>/RAW.log",
  "message": "2 + 2 equals 4.",
  "mode": "line",
  "timestamp": "2026-09-02T03:06:58.123456789Z",
  "offset": 229
}
```

多次 invoke 同一 session 时，RAW.log 以 append 模式累积，每次调用前写入带 task ID 与 query 的分隔 header。`invoke` 响应中的 `raw_log_path` 字段告知前端该文件路径。

## read_cached_content 接口

主动一次性拉取源文件的**全部已有内容**（快照），每次请求都从磁盘最新读取，与采集 offset 无关。开启鉴权后需带 `Authorization: Bearer <token>`，且只能读取当前用户有权限的源/路径：

```powershell
# 按路径（相对或绝对）
curl -H "Authorization: Bearer $tok" "http://127.0.0.1:8080/read_cached_content?path=logs/sample.log"

# 按输入名（返回该输入 glob 命中的文件）
curl -H "Authorization: Bearer $tok" "http://127.0.0.1:8080/read_cached_content?source=prefixed-logs"
```

响应：

```json
{
  "path": "D:/work/.../logs/sample.log",
  "size": 128,
  "returned": 128,
  "truncated": false,
  "content": "20260828-10:00:00.123456 INFO ...\n..."
}
```

- 单文件超 16MB 时只返回末尾 16MB 并置 `truncated: true`。
- `?source=` 命中多个文件时返回 `{"source": "...", "files": ["path1", "path2"]}`，客户端可再按 `path` 逐个拉取。

## 说明

- 轮转检测依赖「新文件大小 < 已读 offset」或文件身份变化（`os.SameFile`，跨平台比较 inode/文件索引），已覆盖主流轮转方式；极端场景（新文件瞬间写满超过旧 offset）轮转后可能漏几行，属预期取舍。
- `tail_files` 默认与 FileBeat 一致为 `false`（首次追踪会回放历史）；若只关心实时增量请设 `true`。