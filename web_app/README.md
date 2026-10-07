# ScienceFlow（Wails 桌面应用）

基于 [Wails v3](https://v3.wails.io) 的 ScienceFlow 桌面控制台。原 `web_app` React + Vite 前端被迁入 `frontend/`，新增 Go 侧代理层负责与后端通信。

## 架构

```
ScienceFlow 桌面应用 (Wails v3)
├── Go 层
│   ├── proxy.go   通用 HTTP 代理 → 转发到日志网关（默认 http://127.0.0.1:8080）
│   ├── gateway.go 日志网关接入 → login/sources/sessions/read_cached_content + SSE 桥接
│   ├── events.go  emitEvent() → 经 app.Event.Emit 推送事件到前端
│   └── app.go     ServiceStartup/ServiceShutdown、gateway URL 运行时配置
└── frontend/      React + Vite 前端（原 web_app）
    ├── src/api/client.ts   通过 bindings 调用 Go（不再直接 fetch）
    ├── src/api/gateway.ts  日志网关客户端（login/会话/流）
    ├── src/hooks/useGatewayStream.ts 通过 Events.On("gateway-log"/"gateway-status") 订阅网关日志流
    └── bindings/           wails3 generate bindings 生成的 Go 绑定（TS）
```

- 所有网关 HTTP 请求由 `api/client.ts` 交给 Go `Proxy()` 方法转发（解决 WebView 跨域）。
- 数据集上传/下载分别走 Go 的 `UploadDataset`/`UploadDatasetFolder`/`DownloadZip`（base64 传输二进制）。

## 日志网关（log_gateway_server）接入

对接 `log_gateway_server`（默认 `http://127.0.0.1:8080`），见 `log_gateway_server/docs/gateway_server_dev.md`：

- **鉴权**：Go `GatewayLogin` → `POST /login`，返回 token 与用户可见源。
- **多端会话**：Go `GatewayCreateSession` / `GatewayUpdateSessionSources` / `GatewayDeleteSession` → `/sessions*`，每个前端连接一个 session，互不影响。
- **流式**：Go `GatewayStartStream(sessionID)` 长连接 `GET /events?session=<id>`，解析 `event: log` 后经 Wails 事件 `gateway-log` 推给前端；连接状态经 `gateway-status`（connecting/connected/disconnected/unauthorized）推送。
- **历史快照**：Go `GatewayReadSnapshot` → `GET /read_cached_content`。
- 前端在 L1「日志」页签通过 `GatewayLogPanel` 登录、订阅源、实时滚动查看；用户名/密码对应 `AUTH.yaml` 中的账号。

## 后端配置

`app_config.yaml`（与可执行文件同目录，其次当前工作目录）：

```yaml
gateway: http://127.0.0.1:8080
```

找不到时代码内置默认 `http://127.0.0.1:8080`。运行时也可通过前端设置里的「日志网关」调用 `SetGatewayURL` 修改。

## 开发

```powershell
wails3 dev
```

> 普通浏览器直接打开 Vite 开发地址时 Wails runtime 不存在，接口会失效；请通过 `wails3 dev` 启动。

## 构建

```powershell
wails3 build
```

产物位于 `bin/scienceflow_gui.exe`；打包安装器（NSIS）用 `wails3 package`。发布时把 `app_config.yaml` 与可执行文件放在同一目录即可覆盖默认网关地址。

## 目录结构

```
web_app/
├── main.go / app.go / proxy.go / gateway.go / events.go   # Go 后端
├── go.mod / go.sum
├── Taskfile.yml                              # wails3 构建入口（build/package/dev）
├── build/                                    # v3 构建资源（Taskfile 分平台、图标、info.json）
├── wails.json                                # Wails 工程配置
├── app_config.yaml                           # gateway 配置
└── frontend/                                 # React + Vite 前端
    └── bindings/                             # wails3 generate bindings 生成的 TS 绑定
```

## 遗留注意事项

- `frontend/public/.env` 仍含真实 DeepSeek API Key，会在构建时被打进 `dist`，需改为运行时注入或后端下发。
- 报告中的图片走 `/api/workspace/file/raw`，桌面端尚无对应代理（见 `ReportViewer.resolveImageSrc`），为已知待办。
