## ScienceFlow 网关服务 + 桌面客户端

本分支（`app`）是 ScienceFlow 的**应用发行分支**，包含两个可独立构建、协同工作的组件：

| 组件 | 目录 | 说明 | 技术栈 |
|---|---|---|---|
| 日志/任务网关 | [`log_gateway_server/`](log_gateway_server/) | 多用户鉴权、会话管理、日志 SSE 实时推流，并负责按需拉起 ScienceFlow 智能体子进程 | Go（单二进制） |
| 桌面控制台 | [`web_app/`](web_app/) | Agent Map、对话轨迹、工作区文件、监控面板 | Wails v3 + React 18 + Vite 6 |

> **ScienceFlow 智能体**（研究框架本体）通过 pip 分发，安装命令：`pip install scienceflow`。
> 网关在启动引导阶段会自动检测并（可选）安装/更新它。

---

## 零基础一键部署（面向非技术人员）

> 不需要编译、不需要懂代码：拿到预编译程序后，按下面几步即可用起来。
> Windows 10/11 已自带所需的 WebView2 运行环境，无需额外安装。

**三个程序分别是什么：**

| 程序 | 作用 | 部署位置 |
|---|---|---|
| `scienceflow_lgw.exe` | **网关服务**：账号登录、运行任务、保存结果，是整个系统的「大脑」 | 必装，最先启动 |
| `scienceflow_gui.exe` | **个人终端版**：单人使用的桌面窗口 | 自己的电脑 |
| `scienceflow_gui-server.exe` | **Web 服务版**：装在服务器上，大家用浏览器访问 | 服务器（可选） |

### 第一步：安装 Python（首次使用）

1. 打开 <https://www.python.org/downloads/>，下载 **Python 3.11 或更高版本**并安装；
2. 安装界面务必勾选 **「Add python.exe to PATH」**；
3. 智能体程序（scienceflow）**无需手动安装**：网关首次启动时会自动检测并询问，直接按回车（默认「是」）即可自动下载安装。

### 第二步：启动网关服务（必须）

新建一个文件夹，放入发布包中的三个文件，双击 `scienceflow_lgw.exe`：

```text
ScienceFlow/                ← 新建文件夹（名称随意）
├── scienceflow_lgw.exe     ← 网关程序（双击启动）
├── config.json             ← 网关配置（默认即可）
└── AUTH.yaml               ← 账号密码文件（见第四步）
```

启动后按屏幕提示操作（全部直接按**回车**即可）：

1. 提示安装 scienceflow → 回车自动安装；
2. 提示「启动端口 `[8901]`」 → 回车使用默认端口 **8901**（记住这个端口号）；
3. 提示端口被占用 → 按提示输入 `y` 结束旧程序，或输入其他端口号；
4. 看到 `listening on http://0.0.0.0:8901` 即启动成功。**窗口不要关闭**（关闭窗口＝停止服务）。

### 第三步：启动终端（二选一）

**方式 A：个人终端版（自己电脑使用）**

1. 新建文件夹，放入 `scienceflow_gui.exe` 和 `app_config.yaml`；
2. 用记事本打开 `app_config.yaml`，把 `gateway` 改成网关地址：
   - 网关在本机：`gateway: http://127.0.0.1:8901`
   - 网关在另一台电脑：`gateway: http://那台电脑的IP:8901`
3. 双击 `scienceflow_gui.exe`，用账号登录（默认管理员 `admin / admin123`）。

**方式 B：Web 服务版（部署在服务器，多人用浏览器访问）**

1. 新建文件夹，放入 `scienceflow_gui-server.exe` 和 `app_config.yaml`；
2. 用记事本编辑 `app_config.yaml`（网关地址 + 网页访问端口）：

```yaml
gateway: http://127.0.0.1:8901   # 网关地址
server:
  host: 0.0.0.0
  port: 8080                     # 浏览器访问端口，按需修改
```

3. 双击启动后，同事在浏览器打开 `http://服务器IP:8080` 即可登录使用；
4. 多人同时使用互不影响，每人用自己的账号登录（见第四步）。

### 第四步：账号与密码（重要）

默认账号（在 `AUTH.yaml` 中）：

```yaml
sciflow: "123456"      # 普通账号

admin:                 # 超级管理员账号（role: admin）
  password: admin123
  role: admin
```

- 用记事本修改 `AUTH.yaml` 后**保存即生效，无需重启**（约 5 秒内自动加载）；
- 用户名只能包含字母、数字、点、下划线和连字符；
- **对外使用前请务必修改默认密码！**

### 常见问题

| 现象 | 处理办法 |
|---|---|
| 双击后窗口一闪而过 | 检查是否被杀毒软件拦截；右键「以管理员身份运行」重试 |
| Windows 提示「已保护你的电脑」 | 点击「更多信息」→「仍要运行」（程序未做数字签名） |
| 首次启动弹出防火墙提示 | 点击「允许访问」（否则其他电脑无法连接） |
| 终端登录失败 / 连不上网关 | 确认网关窗口仍在运行；确认 `app_config.yaml` 的地址、端口与网关提示的端口一致 |
| Web 服务版浏览器打不开 | 确认服务器防火墙放行网页端口（如 8080），浏览器地址以 `http://` 开头 |
| 端口被占用 | 按网关提示确认结束旧进程，或改用其他端口（终端配置需同步修改） |

> 发布包中若没有网关程序 `scienceflow_lgw.exe`，请让技术人员按本文档「二、构建并部署网关服务」编译一次；更详细的配置项见各组件 README。

---

## 架构与数据流

```text
                       pip install scienceflow
  ┌────────────────┐  ◀──────────────────────────────  ┌───────────────────────┐
  │ ScienceFlow    │      子进程: python -m             │  log_gateway_server   │
  │ Agent CLI      │      scienceflow.cli repl          │  ─ 鉴权 / 会话         │
  └───────┬────────┘                                    │  ─ 日志 SSE 推流       │
          │ RAW.log / interaction.log                   │  ─ agent 任务调度      │
          ▼                                             └───────────┬───────────┘
  ┌────────────────┐                                                │ SSE + REST
  │ workspaces/    │  每用户/会话独立工作区                          ▼
  │ task_logs/...  │                                    ┌───────────────────────┐
  └────────────────┘                                    │  web_app (Wails v3)   │
                                                        │  桌面应用 / 网页服务    │
                                                        └───────────────────────┘
```

1. 用户在桌面端（或 server 模式下的浏览器）发送消息，网关在 `workspaces/<user>/<session>/` 下生成运行清单并拉起安装版 ScienceFlow 智能体；
2. 智能体运行过程中的 stdout/stderr 与完整交互记录分别写入 `task_logs/<user>/<session>/RAW.log` 与 `<session>/run/.logs/interaction.log`；
3. 网关监听这两个文件，经 SSE 实时推送给已订阅的桌面端会话；桌面端解析日志重建对话与执行轨迹（Agent Map、消息块、监控面板）。

---

## 环境要求

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python + pip | 3.11+ | 运行 ScienceFlow 智能体（`pip install scienceflow`） |
| Go | 网关 1.22+；桌面端 1.25+ | 编译网关与桌面应用 |
| Node.js + npm | 20 LTS+ | 构建桌面端前端 |
| [Wails v3 CLI](https://v3.wails.io) | `v3.0.0-beta.25` | 构建桌面应用 |
| NSIS（可选） | 任意近期版本 | `wails3 package` 生成 Windows 安装包 |
| WebView2 Runtime | — | 桌面应用运行时（Windows 10/11 通常已内置） |

安装 Wails CLI（版本需与 `web_app/go.mod` 一致）：

```bash
go install github.com/wailsapp/wails/v3/cmd/wails3@v3.0.0-beta.25
```

---

## 一、安装 ScienceFlow 智能体

```bash
# 安装（当前发布通道为预发布版本，建议加 --pre）
pip install scienceflow
# 或：python -m pip install --pre scienceflow
```

验证安装：

```bash
python -c "import scienceflow, importlib.metadata as m; print(m.version('scienceflow'))"
```

网关的**启动引导**会自动完成检测、安装与更新（详见下文 `-bootstrap` / `-yes` 参数）：

```text
◆ [2/4] 检测 scienceflow 安装
  ⚠ 未检测到 scienceflow（包名 scienceflow）

  ⚠ 需要安装 scienceflow 才能运行 agent 任务
  Python        /usr/local/bin/python (3.12.13)
  安装命令      python -m pip install --pre scienceflow
  ? 是否自动安装 scienceflow？ [Y/n]
```

引导最后还会提示**启动端口**（交互式默认 `8901`）并检测端口占用：若被占用会显示占用进程并询问是否中止（详见 [log_gateway_server/README.md](log_gateway_server/README.md#启动引导scienceflow-环境自检)）。

> 若使用自定义解释器或 `uv`，请设置网关配置中的 `agent.python`（如 `"uv run python"`）。

---

## 二、构建并部署网关服务（log_gateway_server）

### 1. 构建

```bash
cd log_gateway_server

# Windows
go build -o scienceflow_lgw.exe .

# Linux（可交叉编译）
GOOS=linux GOARCH=amd64 go build -o scienceflow_lgw .
```

产物为**单一可执行文件**，无运行时依赖（registry 检查点除外，见下）。

### 2. 配置

网关读取 `config.json`（用 `-config` 指定路径），核心配置段：

```jsonc
{
  "server": { "host": "0.0.0.0", "port": 52001, "sse_path": "/events" },
  "auth":   { "enabled": true, "file": "AUTH.yaml", "token_ttl": "24h" },
  "agent": {
    "enabled": true,
    "python": "python",                 // 或 "uv run python"
    "workspace_root": "../workspaces",  // 工作区根目录（也可用环境变量 SCIFLOW_WORKSPACE_ROOT）
    "timeout": "3600s",
    "max_concurrent": 4
  },
  "inputs": [ /* 可选：样例日志源（prefixed-logs / raw-logs / plain-text） */ ],
  "registry": { "path": "./data/registry.json" }
}
```

用户账号位于 `AUTH.yaml`（与网关鉴权、桌面端登录共用；**修改后热更新，无需重启**）：

```yaml
sciflow: "123456"          # 简化写法：可读所有源
# 完整写法（按源授权 + 角色）：
# viewer:
#   password: viewer456
#   sources: ["prefixed-logs", "plain-text"]

admin:                     # 超级管理员：登录后可通过 /users 接口增删改账号
  password: admin123
  role: admin
```

> 超级管理员登录后可用 `/users` 接口在线管理账号（新增 / 改密码 / 改权限 / 删除，即时生效），详见 [log_gateway_server/README.md](log_gateway_server/README.md#控制面接口bearer-token-鉴权)。
> `inputs` 中的样例日志源仅用于演示 SSE 能力；对接智能体任务时必需的是 `agent` 段与 `agent-logs` 源（网关自动接管）。
> **部署前请务必修改 `AUTH.yaml` 中的示例口令。**

### 3. 运行

```bash
# 前台运行（交互式启动引导：检测/安装/更新 scienceflow、选择端口）
./scienceflow_lgw.exe -config config.json

# 后台 / 服务化部署：跳过交互引导
./scienceflow_lgw.exe -config config.json -bootstrap=false

# 无人值守：自动确认安装/更新（不会自动中止端口占用进程）
./scienceflow_lgw.exe -config config.json -yes
```

启动引导要点：

- **端口**：交互式运行会提示 `启动端口 [8901]`，回车采用默认 8901；`-bootstrap=false` / `-yes` / 后台运行使用 `config.json` 的 `server.port`。若沿用配置端口，请在提示时手动输入；**端口须与桌面端 `app_config.yaml` 的 `gateway` 地址一致**。
- **端口占用**：显示占用进程名称、PID 与完整命令，交互式可确认是否中止（默认否）——确认后先 `SIGTERM` 优雅终止（Windows `taskkill`），超时未退出再强制终止，并等待端口真正释放后才继续；拒绝或非交互/`-yes` 模式则打印手动命令并以退出码 1 结束，绝不自动杀进程。
- **安装/更新**：缺失 scienceflow 时确认后自动 `pip install`，拒绝则打印命令并退出；有更新时确认后 `pip install --upgrade`，拒绝则保留当前版本继续。

健康检查：

```bash
curl http://127.0.0.1:52001/healthz
# {"status":"ok","subscribers":0}
```

### 4. 服务化部署

推荐的部署目录结构：

```text
gateway/
├── scienceflow_lgw.exe   # 网关二进制
├── config.json          # 网关配置（端口须与桌面端一致）
├── AUTH.yaml            # 用户账号
├── logs/                # 可选：演示日志源
├── data/registry.json   # 采集检查点（自动生成，建议持久化）
└── workspaces/          # 任务工作区根目录（见 agent.workspace_root）
```

- **Windows**：用 [NSSM](https://nssm.cc/) / WinSW 注册为服务（工作目录设为上述目录，参数 `-config config.json -bootstrap=false`），或 `Start-Process` 后台重定向日志；
- **Linux**：`systemd` 单元或 `nohup ./scienceflow_lgw -config config.json > scienceflow_lgw.out.log 2>&1 &`；
- 重启安全：采集 offset 持久化在 `registry.path`，优雅退出会落盘检查点，重启后无缝续读。

### 5. 生产部署要点

1. `registry.path` 指向持久化目录（容器场景挂载 volume）；
2. `inputs[].path` 使用绝对路径；
3. 对外暴露时经反向代理（Nginx/Caddy）加 TLS 与鉴权；SSE 已内置 `X-Accel-Buffering: no`；
4. 网关为单实例有状态（检查点），多实例共读同一批日志会导致重复推送；
5. **重启恢复**：网关重启后会扫描任务工作区，自动恢复历史会话（会话 ID、名称、历史日志与续聊 memory 均保留）；仅创建过会话但从未执行任务的空会话不会恢复。请持久化 `workspaces/` 与 `task_logs/` 目录。

> 完整配置字段、SSE 协议、控制面 API（`/login`、`/sessions/*`、`/models/*`、`/invoke` 等）见 **[log_gateway_server/README.md](log_gateway_server/README.md)**。

---

## 三、构建并部署桌面应用（web_app）

### 1. 开发模式

```bash
cd web_app
wails3 dev
```

- 热重载；前端源码在 `frontend/`，Go 侧在 `web_app/` 根目录；
- 开发构建脚本位于 `build/`（Wails 标准构建脚手架；该目录未入库，缺失时用 Wails CLI 恢复/初始化）；绑定代码由 `wails3 generate bindings -ts` 自动生成到 `frontend/bindings/`；
- **请勿**直接用浏览器打开 Vite 地址：Wails 运行时绑定不存在，接口不可用。

### 2. 生产构建

```bash
cd web_app
wails3 build            # 产物：bin/scienceflow_gui.exe（Windows）
wails3 package          # 生成 NSIS 安装包（需安装 makensis）
```

`wails3 build` 会依次完成：前端依赖安装 → TypeScript 绑定生成 → 前端构建（`frontend/dist`）→ 图标/版本资源（`build/config.yml` 的 `info` 段）→ Go 编译（`-tags production -H windowsgui`）。

#### 仅构建前端 dist（可选，无需 Wails CLI）

若只需要前端产物（CI 校验、或供其他流程嵌入），可独立构建；产物与 `wails3 build` 内嵌的完全一致（相同 `frontend/bindings`、相同输出目录 `frontend/dist`）：

```bash
cd web_app/frontend
npm ci                 # 或 npm install
npm run build:dist     # → frontend/dist
```

已安装 Wails CLI 时，也可从 `web_app/` 目录使用等价入口：

```bash
wails3 task build:frontend
```

> `frontend/bindings/` 为已入库的生成物，独立构建直接复用；若 Go 侧接口有变更，以 `wails3 build`（会自动重新生成绑定）为准。`frontend/dist` 被 Git 忽略；纯浏览器打开 `dist/index.html` 不可用（无 Wails 运行时，接口与 SSE 均不可达）。

### 3. 部署布局

```text
scienceflow-gui/
├── scienceflow_gui.exe
├── app_config.yaml          # 网关地址（与 exe 同目录，其次当前工作目录）
└── themes/
    └── backgrounds/         # 可选：Agent 地图背景图（文件名即设置中的选项名）
```

`app_config.yaml`：

```yaml
gateway: http://127.0.0.1:52001   # 必须与网关 config.json 的 server.port 一致
```

- 未找到配置文件时使用内置默认 `http://127.0.0.1:52001`，也可在应用内设置或登录页的「网关」栏运行时修改；
- `themes/backgrounds/` 目录在首次运行时自动创建，放入 PNG/JPG/WebP 等图片即可在「设置 → 背景图片」中选择；未放置时使用内置默认背景；
- 登录账号与网关 `AUTH.yaml` 一致（如示例 `admin / admin123`）。

### 4. 验证

1. 启动网关，`curl http://127.0.0.1:52001/healthz` 返回 `ok`；
2. 启动桌面应用，登录后打开「设置 → 测试连接」：仅探测网关连通性，顶栏状态指示同步显示 `已连接`；
3. 在对话面板发送消息，网关拉起智能体后，Agent Map、执行轨迹与「日志」页签将实时回放运行过程。

### 5. 网页远程服务（server 模式，可选）

同一份源码可构建为**纯 HTTP 服务**（无 GUI / WebView 依赖），部署在服务器上供浏览器远程访问：

```bash
# 一键：生产构建 + 启动（开发机验证用）
cd web_app
wails3 task serve

# 仅构建（产物 bin/scienceflow_gui-server.exe）
wails3 task build:server

# Linux 交叉编译（推荐：先自动构建前端 + bindings，再产出 linux/amd64 二进制）
# 产物 bin/scienceflow_gui-server-linux-amd64；如需 arm64 追加 LINUX_ARCH=arm64
wails3 task build:server:linux

# 手动等价流程（两步都必须执行——前端 dist 是 go:embed 打进去的，
# 直接 go build 会嵌入过期的 frontend/dist）
cd frontend && npm run build:dist && cd ..
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -tags server,production -trimpath -buildvcs=false -ldflags="-w -s" -o bin/scienceflow_gui-server-linux-amd64 .
```

服务器部署布局：

```text
scienceflow-web/
├── scienceflow_gui-server-linux-amd64   # server 模式二进制
├── app_config.yaml               # 网关地址 + 网页服务监听地址
└── themes/backgrounds/           # 可选：背景图
```

`app_config.yaml` 中新增可选的 `server` 段（缺省 `0.0.0.0:8080`，也可用环境变量 `WAILS_SERVER_HOST` / `WAILS_SERVER_PORT` 覆盖）：

```yaml
gateway: http://127.0.0.1:52001
server:
  host: 0.0.0.0
  port: 8080
```

- 启动后浏览器访问 `http://<服务器IP>:8080/`，登录账号仍为网关 `AUTH.yaml` 账号；
- **网关地址由服务端解析**（`app_config.yaml` 的 `gateway`，查找顺序：二进制同目录 → 工作目录），浏览器不直接访问网关；启动日志会打印生效配置，例如 `ScienceFlow (server): gateway=http://127.0.0.1:52001 server(config)=0.0.0.0:8080`；
- 浏览器内的所有调用（网关代理、文件、SSE 日志流）均由服务端进程完成，客户端无需安装任何组件；多用户可同时访问（各自独立登录/会话）；
- 本机资源统计：Windows 用系统计数器、Linux 用 `/proc/stat`、`/proc/meminfo` 与根分区 `statfs` 采集（CPU/内存/磁盘实时展示）；其他平台显示为 0（不采集），任务与日志链路不受影响；
- **安全提示**：默认监听所有网卡，请置于内网并配合防火墙/反向代理（HTTPS + 认证）；数据访问控制由网关登录完成；
- 容器化（可选）：`wails3 task build:docker` / `wails3 task run:docker`（脚手架内置，容器内默认端口 8080）；
- 若日志出现 `... got HTML response from <地址> ...`：说明该地址返回的是网页而非日志网关（多为 `app_config.yaml` 缺失或指向了别的 Web 服务），请修正 `gateway`，或直接在登录页「网关」栏修改。

---

## 四、端到端快速开始（Linux 示例）

```bash
# 1) 安装智能体
python -m pip install --pre scienceflow

# 2) 构建并启动网关（-yes：无人值守，自动校验/更新 scienceflow，使用 config.json 端口 52001）
cd log_gateway_server
go build -o scienceflow_lgw .
./scienceflow_lgw -config config.json -yes &
curl http://127.0.0.1:52001/healthz

# 3) 构建桌面应用并部署
cd ../web_app
wails3 build
mkdir -p dist && cp bin/scienceflow_gui.exe dist/
cp app_config.yaml dist/
# 将 dist/ 拷贝到目标机器运行（Windows 需 WebView2 Runtime）
```

---

## 仓库结构

```text
.
├── log_gateway_server/        # 网关服务（Go）
│   ├── main.go                # 入口：config / registry / hub / tailer / server 装配
│   ├── internal/              # auth / session / agent / tailer / server / hub ...
│   ├── config.json            # 示例配置（端口 52001）
│   └── AUTH.yaml              # 示例账号
├── web_app/                   # 桌面应用（Wails v3）
│   ├── main.go / app.go / gateway.go / proxy.go / themes.go ...
│   ├── build/                 # Wails 构建脚手架（Taskfile、图标、info.json）
│   ├── frontend/              # React + Vite 前端（src/、bindings/）
│   ├── app_config.yaml        # 网关地址
│   └── Taskfile.yml           # build / dev / package 入口
├── workspaces/                # 运行期任务工作区（不入库）
└── data/                      # 运行期数据（不入库）
```

## 测试

```bash
cd log_gateway_server && go test ./...
cd ../web_app && go test ./...          # themes 解析等单元测试
cd frontend && npm run typecheck        # 前端类型检查
```

## 相关文档

- [log_gateway_server/README.md](log_gateway_server/README.md)：网关完整配置、鉴权与会话模型、SSE 消息格式、agent 调度与 `/models` registry 对接；
- [web_app/README.md](web_app/README.md)：桌面端架构与实现细节；
- ScienceFlow 框架源码与论文：<https://arxiv.org/abs/2608.14354>。

## License

[MIT License](LICENSE)
