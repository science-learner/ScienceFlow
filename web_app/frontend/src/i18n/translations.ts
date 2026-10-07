export type Lang = 'zh-CN' | 'en-US';

export interface Translations {
  topbar: {
    title: string;
    statusConnected: string;
    statusConnecting: string;
    statusDisconnected: string;
    statusUnauthorized: string;
    workspaceReady: string;
    batch: string;
    settings: string;
    backToAgentMap: string;
    switchToChinese: string;
    switchToEnglish: string;
    logout: string;
  };
  login: {
    tagline1: string;
    tagline2: string;
    authRequired: string;
    usernamePlaceholder: string;
    passwordPlaceholder: string;
    usernameRequired: string;
    passwordRequired: string;
    signIn: string;
    welcome: string;
    loginFailed: string;
    cachedHint: string;
    engineName: string;
  };
  agentMap: {
    imgAlt: string;
    openBoard: string;
    openBoardAria: string;
    keyReport: string;
    keyReportAria: string;
    needsAttention: string;
    openIssue: string;
  };
  batchPanel: {
    batchRailLabel: string;
    currentBatch: string;
    runs: string;
    bestEffective: string;
    elapsed: string;
    dimFieldIs: string;
  };
  chatRail: {
    cockpit: string;
    sessions: string;
    chatSessions: string;
    refresh: string;
    createTask: string;
    workspaceName: string;
    workspaceNamePlaceholder: string;
    create: string;
    cancel: string;
    createsLiteWorkspace: string;
    messages: string;
    lite: string;
    detail: string;
    send: string;
    collapsed: string;
    inputPlaceholder: string;
    taskSetup: string;
    ready: string;
    noTask: string;
    createTaskFirst: string;
    mode: string;
    heavy: string;
    dataset: string;
    uploadFolder: string;
    taskSetupTitle: string;
    listening: string;
    starting: string;
    you: string;
    system: string;
    agent: string;
    toolTrace: string;
    running: string;
    stopping: string;
    stopped: string;
    done: string;
    resolved: string;
    expired: string;
    pendingDecision: string;
    preparingUpload: string;
    uploading: string;
    uploadingFolder: string;
    complete: string;
    unknownError: string;
    chat: string;
    chatModeLabel: string;
    agentModeLabel: string;
    selectChatMode: string;
    uploadFile: string;
    selectAgent: string;
    noDataset: string;
    runLabel: string;
    toolCallsUnit: string;
    iterationsShort: string;
    reasoning: string;
    output: string;
    agentWorking: string;
    agentStarting: string;
    copyAnswer: string;
    copied: string;
    emptyTitle: string;
    emptyHint: string;
    emptyOfflineHint: string;
  };
  l1Workspace: {
    board: string;
    lineage: string;
    workspace: string;
    logs: string;
    monitor: string;
    monitorSubtitle: string;
    taskScope: string;
    noReadme: string;
    status: string;
    bestMetric: string;
    stages: string;
    elapsed: string;
    node: string;
    nodes: string;
    runs: string;
    taskBoard: string;
    taskStatus: string;
    noMonitorData: string;
    loadingMonitor: string;
    collapse: string;
    expand: string;
    runtimeStatus: string;
    control: string;
    codeAgent: string;
    deepAgent: string;
    ensemble: string;
    scheduler: string;
    phase: string;
    exploitExplore: string;
    agentMapTab: string;
    latencyLabel: string;
    throughputLabel: string;
    jobs: string;
    active: string;
    failed: string;
    monitorFile: string;
    budget: string;
    wall: string;
    left: string;
    budgetRemaining: string;
    riskHints: string;
    activeAlerts: string;
    noActiveAlerts: string;
    costAndLatency: string;
    cpu: string;
    memory: string;
    storage: string;
    cost: string;
    tokensReserved: string;
    reserved: string;
    queue: string;
    queued: string;
    done: string;
    total: string;
    cacheHit: string;
    promptReuseWeighted: string;
    tokensIn: string;
    tokensOut: string;
    costHook: string;
    pendingApiUsage: string;
    latencyHealth: string;
    responsive: string;
    latencyHealthyDesc: string;
    firstResponse: string;
    avgMax: string;
    fullReply: string;
    slowestCall: string;
    worstRecent: string;
    samples: string;
    recentWindow: string;
    recentLatencySamples: string;
    tokens: string;
    cache: string;
    nextMonitorAction: string;
    noLatencySamples: string;
    researchStages: string;
    searchGraph: string;
    noNodes: string;
    nodeId: string;
    created: string;
    unableToLoadFile: string;
    selectTextFile: string;
    fileExtensions: string;
    files: string;
    unableToLoadLog: string;
    noLogs: string;
    selectLogFile: string;
    logsLive: string;
    logsHistory: string;
    logsClear: string;
    logsWaiting: string;
    running: string;
    file: string;
    copyContent: string;
    download: string;
    taskState: string;
    exitCode: string;
    rounds: string;
    modelsLabel: string;
    activity: string;
    toolsLabel: string;
    failuresLabel: string;
    llmCallsLabel: string;
    turnsLabel: string;
    contextAndMemory: string;
    contextLength: string;
    contextMessages: string;
    compactions: string;
    contextTriggers: string;
    longTermMemory: string;
    shortTermMemory: string;
    efficiencyAndRisk: string;
    outputCompression: string;
    writesEdits: string;
    metricProduced: string;
    riskNone: string;
    recentCalls: string;
    modelLabel: string;
    roleLabel: string;
    cacheRateLabel: string;
    tpotLabel: string;
    finishLabel: string;
    localResources: string;
  };
  reportViewer: {
    reportNav: string;
    keyReport: string;
    download: string;
    pdf: string;
    scienceflow: string;
    defaultLead: string;
  };
  sessionPanel: {
    chatSessions: string;
    close: string;
    newSession: string;
    delete: string;
    confirm: string;
    cancel: string;
    confirmDelete: string;
    loading: string;
    noSessions: string;
    messages: string;
    last: string;
    lite: string;
    refresh: string;
    switchFailed: string;
  };
  settingsModal: {
    settings: string;
    done: string;
    display: string;
    theme: string;
    dark: string;
    paperLight: string;
    textSize: string;
    backgroundImage: string;
    agentMapBackground: string;
    small: string;
    default: string;
    large: string;
    workspace: string;
    baseFolder: string;
    baseFolderPlaceholder: string;
    model: string;
    deepseekV4Flash: string;
    modelName: string;
    modelNamePlaceholder: string;
    coderModel: string;
    feedbackModel: string;
    apiKey: string;
    apiKeyPlaceholder: string;
    apiUrl: string;
    apiUrlPlaceholder: string;
    keepApiKey: string;
    modelManagement: string;
    provider: string;
    operation: string;
    builtinModels: string;
    customModels: string;
    customProvider: string;
    addModel: string;
    addingModel: string;
    editModel: string;
    updateModel: string;
    updatingModel: string;
    deleteModel: string;
    deletingModel: string;
    confirmDeleteModel: string;
    modelAdded: string;
    modelUpdated: string;
    modelDeleted: string;
    activating: string;
    modelActivated: string;
    loadingModels: string;
    noModels: string;
    close: string;
    reset: string;
    test: string;
    saving: string;
    save: string;
    testing: string;
    connectionOk: string;
    connectionFailed: string;
    saved: string;
    unknownError: string;
    ready: string;
  };
  statePanel: {
    stateDetails: string;
    close: string;
    reset: string;
    transport: string;
    noTransport: string;
    scheduler: string;
    noScheduler: string;
    moduleEtags: string;
    noEtags: string;
    workspace: string;
    noTaskRoot: string;
    dimFieldIs: string;
  };
  common: {
    dash: string;
    unknown: string;
    attached: string;
    idle: string;
    hundredPercent: string;
    newSession: string;
  };
}

export const zhCN: Translations = {
  topbar: {
    title: 'ScienceFlow',
    statusConnected: '已连接',
    statusConnecting: '连接中',
    statusDisconnected: '未连接',
    statusUnauthorized: '未授权',
    workspaceReady: '工作空间就绪',
    batch: '批次',
    settings: '设置',
    backToAgentMap: '返回Agent地图',
    switchToChinese: '切换到中文',
    switchToEnglish: '切换到英文',
    logout: '退出登录',
  },
  login: {
    tagline1: '让长程研究',
    tagline2: '可恢复 · 可适应 · 高效',
    authRequired: '需要身份验证',
    usernamePlaceholder: '用户名',
    passwordPlaceholder: '密码',
    usernameRequired: '请输入用户名',
    passwordRequired: '请输入密码',
    signIn: '登录',
    welcome: '欢迎使用 ScienceFlow',
    loginFailed: '登录失败',
    cachedHint: '登录状态缓存 7 天',
    engineName: 'ScienceFlow 智能体引擎',
  },
  agentMap: {
    imgAlt: 'Agent地图工作空间',
    openBoard: '打开Agent工作空间',
    openBoardAria: '打开L1面板',
    keyReport: '关键报告',
    keyReportAria: '打开关键报告',
    needsAttention: '需要关注',
    openIssue: '打开问题',
  },
  batchPanel: {
    batchRailLabel: '批次',
    currentBatch: '当前批次',
    runs: '运行',
    bestEffective: '最佳有效值',
    elapsed: '耗时',
    dimFieldIs: '为',
  },
  chatRail: {
    cockpit: '驾驶舱',
    sessions: '会话',
    chatSessions: '聊天会话',
    refresh: '刷新',
    createTask: '创建任务',
    workspaceName: '工作空间名称',
    workspaceNamePlaceholder: '可选任务文件夹名称',
    create: '创建',
    cancel: '取消',
    createsLiteWorkspace: '正在创建轻量工作空间...',
    messages: '条消息',
    lite: 'lite',
    detail: '详情',
    send: '发送',
    collapsed: '已折叠',
    inputPlaceholder: '请输入您的问题（shift + enter换行）',
    taskSetup: '任务设置',
    ready: '就绪',
    noTask: '无任务',
    createTaskFirst: '请先创建一个任务。',
    mode: '模式',
    heavy: 'heavy',
    dataset: '数据集',
    uploadFolder: '上传文件夹',
    taskSetupTitle: '任务设置',
    listening: '正在聆听',
    starting: '启动中...',
    you: '你',
    system: '系统',
    agent: '智能体',
    toolTrace: '工具追踪',
    running: '运行中',
    stopping: '停止中',
    stopped: '已停止',
    done: '完成',
    resolved: '已解决',
    expired: '已过期',
    pendingDecision: '待处理决策',
    preparingUpload: '准备上传',
    uploading: '上传中...',
    uploadingFolder: '上传文件夹...',
    complete: '完成',
    unknownError: '未知错误',
    chat: '聊天',
    chatModeLabel: '聊天模式',
    agentModeLabel: '智能体模式',
    selectChatMode: '选择对话模式',
    uploadFile: '上传文件',
    selectAgent: '请在地图上选择一个智能体。',
    noDataset: '未挂载数据集。',
    runLabel: '运行',
    toolCallsUnit: '次工具调用',
    iterationsShort: '迭代',
    reasoning: '思考',
    output: '输出',
    agentWorking: '智能体执行中…',
    agentStarting: '启动中',
    copyAnswer: '复制回答',
    copied: '已复制',
    emptyTitle: 'Agent Copilot',
    emptyHint: '发送消息后，智能体的思考、工具调用与答复将在此实时呈现。',
    emptyOfflineHint: '连接日志网关后，智能体的工作流将在此实时呈现。',
  },
  l1Workspace: {
    board: '看板',
    lineage: '谱系',
    workspace: '工作空间',
    logs: '日志',
    monitor: '监控',
    monitorSubtitle: '资源 · 文件 · 控制',
    taskScope: '任务范围',
    noReadme: '无可用说明',
    status: '状态',
    bestMetric: '最佳指标',
    stages: '阶段',
    elapsed: '耗时',
    node: '节点',
    nodes: '节点数',
    runs: '运行次数',
    taskBoard: '任务看板',
    taskStatus: '任务状态:',
    noMonitorData: '无可用监控数据',
    loadingMonitor: '正在加载监控数据...',
    collapse: '收起',
    expand: '展开',
    runtimeStatus: '运行时状态',
    control: '控制',
    codeAgent: '代码智能体',
    deepAgent: '深度智能体',
    copyContent: '复制内容',
    download: '下载',
    ensemble: '集成',
    scheduler: '调度器',
    phase: '阶段',
    exploitExplore: 'exploit_explore',
    agentMapTab: 'Agent地图',
    latencyLabel: '延迟',
    throughputLabel: '吞吐',
    jobs: '作业',
    active: '活跃',
    failed: '失败',
    monitorFile: '数据源',
    budget: '预算',
    wall: '墙上时间',
    left: '剩余',
    budgetRemaining: '预算剩余',
    riskHints: '风险提示',
    activeAlerts: '条活跃告警',
    noActiveAlerts: '无活跃告警',
    costAndLatency: '资源消耗',
    cpu: 'CPU（系统）',
    memory: '内存（系统）',
    storage: '存储（系统）',
    cost: '成本',
    tokensReserved: 'Tokens',
    reserved: '预留',
    queue: '队列',
    queued: '排队中',
    done: '完成',
    total: '总计',
    cacheHit: '缓存命中',
    promptReuseWeighted: '提示词复用加权',
    tokensIn: '输入Token',
    tokensOut: '输出Token',
    costHook: '成本钩子',
    pendingApiUsage: '待处理API用量连接器',
    latencyHealth: '延迟健康度',
    responsive: '响应正常',
    latencyHealthyDesc: '最近的聊天和模型调用处于健康范围内。',
    firstResponse: '首次响应',
    avgMax: '平均 / 最大',
    fullReply: '完整回复',
    slowestCall: '最慢调用',
    worstRecent: '最近最差',
    samples: '样本',
    recentWindow: '最近窗口',
    recentLatencySamples: '最近延迟样本',
    tokens: 'Token',
    cache: '缓存',
    nextMonitorAction: '下次监控操作',
    noLatencySamples: '尚无延迟样本',
    researchStages: '研究阶段',
    searchGraph: '搜索图谱',
    noNodes: '无可用节点',
    nodeId: '节点ID',
    created: '创建时间',
    unableToLoadFile: '无法加载文件',
    selectTextFile: '请选择一个文本文件',
    fileExtensions: '.py · .json · .md',
    files: '文件',
    unableToLoadLog: '无法加载日志',
    noLogs: '无可用日志',
    selectLogFile: '请选择一个日志文件查看',
    logsLive: '实时',
    logsHistory: '历史',
    logsClear: '清空',
    logsWaiting: '等待 Agent 运行…',
    running: '运行中',
    file: '文件',
    taskState: '任务状态',
    exitCode: '退出码',
    rounds: '轮次',
    modelsLabel: '模型',
    activity: '活动',
    toolsLabel: '工具',
    failuresLabel: '失败',
    llmCallsLabel: 'LLM 调用',
    turnsLabel: '对话轮次',
    contextAndMemory: '上下文与内存',
    contextLength: '上下文长度',
    contextMessages: '上下文消息数',
    compactions: '上下文压缩',
    contextTriggers: '上下文触发',
    longTermMemory: '长期记忆',
    shortTermMemory: '短期记忆',
    efficiencyAndRisk: '效率与风险',
    outputCompression: '工具输出压缩',
    writesEdits: '文件写/改',
    metricProduced: '指标产出',
    riskNone: '无风险提示',
    recentCalls: '最近 LLM 调用',
    modelLabel: '模型',
    roleLabel: '角色',
    cacheRateLabel: '缓存率',
    tpotLabel: 'TPOT',
    finishLabel: '结束原因',
    localResources: '本地资源',
  },
  reportViewer: {
    reportNav: '报告导航',
    keyReport: '关键报告',
    download: '下载',
    pdf: '打印PDF',
    scienceflow: 'ScienceFlow',
    defaultLead: 'ScienceFlow 工作空间报告。',
  },
  sessionPanel: {
    chatSessions: '聊天会话',
    close: '关闭',
    newSession: '新建会话',
    delete: '删除',
    confirm: '确认删除',
    cancel: '取消',
    confirmDelete: '再次点击确认删除该会话',
    loading: '加载中...',
    noSessions: '未找到会话',
    messages: '条消息',
    last: '最后:',
    lite: '轻量',
    refresh: '刷新',
    switchFailed: '会话切换失败',
  },
  settingsModal: {
    settings: '设置',
    done: '完成',
    display: '显示',
    theme: '主题',
    dark: '深色',
    paperLight: '纸张白',
    textSize: '文字大小',
    backgroundImage: '背景图片',
    agentMapBackground: '默认Agent地图',
    small: '小',
    default: '默认',
    large: '大',
    workspace: '工作空间',
    baseFolder: '基础目录',
    baseFolderPlaceholder: '/path/to/tasks',
    model: '模型',
    deepseekV4Flash: 'deepseek-v4-flash',
    modelName: '模型名称',
    modelNamePlaceholder: '请输入模型名称',
    coderModel: 'Coder 模型',
    feedbackModel: 'Feedbacker 模型',
    apiKey: 'API Key',
    apiKeyPlaceholder: '请输入 API Key',
    apiUrl: '模型路径',
    apiUrlPlaceholder: '请输入模型路径',
    keepApiKey: '留空则保留当前 API Key',
    modelManagement: '模型管理',
    provider: '模型路径',
    operation: '操作',
    builtinModels: '内置',
    customModels: '自定义',
    customProvider: '自定义',
    addModel: '新增模型',
    addingModel: '新增中...',
    editModel: '编辑模型',
    updateModel: '更新模型',
    updatingModel: '更新中...',
    deleteModel: '删除模型',
    deletingModel: '删除中...',
    confirmDeleteModel: '确定删除这个模型吗？',
    modelAdded: '模型已新增',
    modelUpdated: '模型已更新',
    modelDeleted: '模型已删除',
    activating: '切换中...',
    modelActivated: '模型已切换，新任务生效',
    loadingModels: '加载模型中...',
    noModels: '暂无模型',
    close: '关闭',
    reset: '重置',
    test: '测试连接',
    saving: '保存中...',
    save: '保存',
    testing: '测试中...',
    connectionOk: '连接正常',
    connectionFailed: '连接失败',
    saved: '已保存',
    unknownError: '未知错误',
    ready: '就绪',
  },
  statePanel: {
    stateDetails: '状态详情',
    close: '关闭',
    reset: '重置',
    transport: '传输',
    noTransport: '无传输数据',
    scheduler: '调度器',
    noScheduler: '无调度器数据',
    moduleEtags: '模块Etags',
    noEtags: '无可用etags',
    workspace: '工作空间',
    noTaskRoot: '无活跃任务根目录',
    dimFieldIs: '为',
  },
  common: {
    dash: '—',
    unknown: '未知',
    attached: '已挂载',
    idle: '空闲',
    hundredPercent: '100%',
    newSession: '新会话',
  },
};

export const enUS: Translations = {
  topbar: {
    title: 'ScienceFlow',
    statusConnected: 'Connected',
    statusConnecting: 'Connecting',
    statusDisconnected: 'Disconnected',
    statusUnauthorized: 'Unauthorized',
    workspaceReady: 'Workspace ready',
    batch: 'Batch',
    settings: 'Settings',
    backToAgentMap: 'Back to Agent Map',
    switchToChinese: 'Switch to Chinese',
    switchToEnglish: 'Switch to English',
    logout: 'Sign out',
  },
  login: {
    tagline1: 'Making Long-Horizon Research',
    tagline2: 'Recoverable, Adaptive, and Efficient',
    authRequired: 'Authentication Required',
    usernamePlaceholder: 'Username',
    passwordPlaceholder: 'Password',
    usernameRequired: 'Please enter your username',
    passwordRequired: 'Please enter your password',
    signIn: 'Sign In',
    welcome: 'Welcome to ScienceFlow',
    loginFailed: 'Login failed',
    cachedHint: 'Login state cached for 7 days',
    engineName: 'ScienceFlow Agent Engine',
  },
  agentMap: {
    imgAlt: 'Agent map workspace',
    openBoard: 'Open Agent Workspace',
    openBoardAria: 'Open L1 Board',
    keyReport: 'Key Report',
    keyReportAria: 'Open Key Report',
    needsAttention: 'Needs attention',
    openIssue: 'Open issue',
  },
  batchPanel: {
    batchRailLabel: 'Batch',
    currentBatch: 'Current Batch',
    runs: 'Runs',
    bestEffective: 'best effective',
    elapsed: 'elapsed',
    dimFieldIs: 'is',
  },
  chatRail: {
    cockpit: 'Cockpit',
    sessions: 'Sessions',
    chatSessions: 'Chat Sessions',
    refresh: 'Refresh',
    createTask: 'Create Task',
    workspaceName: 'workspace name',
    workspaceNamePlaceholder: 'optional task folder name',
    create: 'Create',
    cancel: 'Cancel',
    createsLiteWorkspace: 'Creates a Lite workspace...',
    messages: 'messages',
    lite: 'lite',
    detail: 'Detail',
    send: 'Send',
    collapsed: 'collapsed',
    inputPlaceholder: 'Type your message here (shift + enter to change line)',
    taskSetup: 'Task Setup',
    ready: 'Ready',
    noTask: 'No task',
    createTaskFirst: 'Create a task first.',
    mode: 'Mode',
    heavy: 'Heavy',
    dataset: 'Dataset',
    uploadFolder: 'Upload Folder',
    taskSetupTitle: 'Task setup',
    listening: 'Listening',
    starting: 'Starting...',
    you: 'You',
    system: 'System',
    agent: 'Agent',
    toolTrace: 'TOOL TRACE',
    running: 'running',
    stopping: 'stopping',
    stopped: 'stopped',
    done: 'done',
    resolved: 'Resolved',
    expired: 'Expired',
    pendingDecision: 'Pending decision',
    preparingUpload: 'Preparing upload',
    uploading: 'Uploading...',
    uploadingFolder: 'Uploading folder...',
    complete: 'Complete',
    unknownError: 'Unknown error',
    chat: 'Chat',
    chatModeLabel: 'Chat Mode',
    agentModeLabel: 'Agent Mode',
    selectChatMode: 'Select chat mode',
    uploadFile: 'Upload file',
    selectAgent: 'Select an agent on the map.',
    noDataset: 'No dataset attached.',
    runLabel: 'Run',
    toolCallsUnit: 'tool calls',
    iterationsShort: 'iter',
    reasoning: 'Reasoning',
    output: 'Output',
    agentWorking: 'Agent working…',
    agentStarting: 'starting',
    copyAnswer: 'Copy answer',
    copied: 'Copied',
    emptyTitle: 'Agent Copilot',
    emptyHint: 'Send a message — the agent\'s reasoning, tool calls and answer will stream here in real time.',
    emptyOfflineHint: 'Connect to the log gateway to watch the agent workflow in real time.',
  },
  l1Workspace: {
    board: 'Board',
    lineage: 'Lineage',
    workspace: 'Workspace',
    logs: 'Logs',
    monitor: 'Monitor',
    monitorSubtitle: 'resources · files · controls',
    taskScope: 'Task scope',
    noReadme: 'No readme available',
    status: 'Status',
    bestMetric: 'Best Metric',
    stages: 'Stages',
    elapsed: 'Elapsed',
    node: 'Node',
    nodes: 'Nodes',
    runs: 'Runs',
    taskBoard: 'Task Board',
    taskStatus: 'Task status:',
    noMonitorData: 'No monitor data available',
    loadingMonitor: 'Loading monitor data...',
    collapse: 'Collapse',
    expand: 'Expand',
    runtimeStatus: 'Runtime Status',
    control: 'control',
    codeAgent: 'code_agent',
    deepAgent: 'deep_agent',
    copyContent: 'Copy content',
    download: 'Download',
    ensemble: 'ensemble',
    scheduler: 'Scheduler',
    phase: 'phase',
    exploitExplore: 'exploit_explore',
    agentMapTab: 'Agent Map',
    latencyLabel: 'latency',
    throughputLabel: 'throughput',
    jobs: 'jobs',
    active: 'active',
    failed: 'failed',
    monitorFile: 'Source',
    budget: 'Budget',
    wall: 'wall',
    left: 'left',
    budgetRemaining: 'budget remaining',
    riskHints: 'Risk Hints',
    activeAlerts: 'active alerts',
    noActiveAlerts: 'no active alerts',
    costAndLatency: 'Resource Usage',
    cpu: 'CPU (System)',
    memory: 'Memory (System)',
    storage: 'Storage (System)',
    cost: 'Cost',
    tokensReserved: 'Tokens',
    reserved: 'Reserved',
    queue: 'Queue',
    queued: 'queued',
    done: 'Done',
    total: 'total',
    cacheHit: 'cache hit',
    promptReuseWeighted: 'prompt reuse weighted',
    tokensIn: 'tokens in',
    tokensOut: 'tokens out',
    costHook: 'cost hook',
    pendingApiUsage: 'pending API usage connector',
    latencyHealth: 'Latency Health',
    responsive: 'Responsive',
    latencyHealthyDesc: 'Recent chat and model calls are returning in a healthy range.',
    firstResponse: 'first response',
    avgMax: 'avg / max',
    fullReply: 'full reply',
    slowestCall: 'slowest call',
    worstRecent: 'worst recent',
    samples: 'samples',
    recentWindow: 'recent window',
    recentLatencySamples: 'Recent Latency Samples',
    tokens: 'tokens',
    cache: 'cache',
    nextMonitorAction: 'next monitor action',
    noLatencySamples: 'No latency samples yet',
    researchStages: 'Research Stages',
    searchGraph: 'search graph',
    noNodes: 'No nodes available',
    nodeId: 'Node ID',
    created: 'Created',
    unableToLoadFile: 'Unable to load file',
    selectTextFile: 'Select a text file',
    fileExtensions: '.py · .json · .md',
    files: 'Files',
    unableToLoadLog: 'Unable to load log',
    noLogs: 'No logs available',
    selectLogFile: 'Select a log file to view',
    logsLive: 'Live',
    logsHistory: 'History',
    logsClear: 'Clear',
    logsWaiting: 'Waiting for agent run…',
    running: 'running',
    file: 'file',
    taskState: 'Task state',
    exitCode: 'Exit code',
    rounds: 'Rounds',
    modelsLabel: 'Models',
    activity: 'Activity',
    toolsLabel: 'Tools',
    failuresLabel: 'Failures',
    llmCallsLabel: 'LLM calls',
    turnsLabel: 'Turns',
    contextAndMemory: 'Context & Memory',
    contextLength: 'Context length',
    contextMessages: 'Context messages',
    compactions: 'Compactions',
    contextTriggers: 'Context checks',
    longTermMemory: 'Long-term memory',
    shortTermMemory: 'Short-term memory',
    efficiencyAndRisk: 'Efficiency & Risk',
    outputCompression: 'Output compression',
    writesEdits: 'Writes / edits',
    metricProduced: 'Metric produced',
    riskNone: 'No risks detected',
    recentCalls: 'Recent LLM calls',
    modelLabel: 'Model',
    roleLabel: 'Role',
    cacheRateLabel: 'Cache',
    tpotLabel: 'TPOT',
    finishLabel: 'Finish',
    localResources: 'Local machine',
  },
  reportViewer: {
    reportNav: 'Report navigation',
    keyReport: 'Key Report',
    download: 'Download',
    pdf: 'Print PDF',
    scienceflow: 'ScienceFlow',
    defaultLead: 'ScienceFlow workspace report.',
  },
  sessionPanel: {
    chatSessions: 'Chat Sessions',
    close: 'Close',
    newSession: 'New Session',
    delete: 'Delete',
    confirm: 'Confirm Delete',
    cancel: 'Cancel',
    confirmDelete: 'Click again to confirm deletion',
    loading: 'Loading...',
    noSessions: 'No sessions found',
    messages: 'messages',
    last: 'Last:',
    lite: 'lite',
    refresh: 'Refresh',
    switchFailed: 'Failed to switch session',
  },
  settingsModal: {
    settings: 'Settings',
    done: 'Done',
    display: 'Display',
    theme: 'Theme',
    dark: 'Dark',
    paperLight: 'Paper Light',
    textSize: 'Text size',
    backgroundImage: 'Background image',
    agentMapBackground: 'Default Agent Map',
    small: 'Small',
    default: 'Default',
    large: 'Large',
    workspace: 'Workspace',
    baseFolder: 'Base folder',
    baseFolderPlaceholder: '/path/to/tasks',
    model: 'Model',
    deepseekV4Flash: 'deepseek-v4-flash',
    modelName: 'Model name',
    modelNamePlaceholder: 'Enter model name',
    coderModel: 'Coder Model',
    feedbackModel: 'Feedbacker Model',
    apiKey: 'API Key',
    apiKeyPlaceholder: 'Enter API key',
    apiUrl: 'Model Url',
    apiUrlPlaceholder: 'Enter model url',
    keepApiKey: 'Leave blank to keep current API key',
    modelManagement: 'Model management',
    provider: 'Model Url',
    operation: 'Operation',
    builtinModels: 'Built-in',
    customModels: 'Custom',
    customProvider: 'Custom',
    addModel: 'Add model',
    addingModel: 'Adding...',
    editModel: 'Edit model',
    updateModel: 'Update model',
    updatingModel: 'Updating...',
    deleteModel: 'Delete model',
    deletingModel: 'Deleting...',
    confirmDeleteModel: 'Delete this model?',
    modelAdded: 'Model added',
    modelUpdated: 'Model updated',
    modelDeleted: 'Model deleted',
    activating: 'Activating...',
    modelActivated: 'Model switched; applies to new tasks',
    loadingModels: 'Loading models...',
    noModels: 'No models',
    close: 'Close',
    reset: 'Reset',
    test: 'Test Connection',
    saving: 'Saving...',
    save: 'Save',
    testing: 'Testing...',
    connectionOk: 'Connection OK',
    connectionFailed: 'Connection failed',
    saved: 'Saved',
    unknownError: 'Unknown error',
    ready: 'Ready',
  },
  statePanel: {
    stateDetails: 'State Details',
    close: 'Close',
    reset: 'Reset',
    transport: 'Transport',
    noTransport: 'No transport data',
    scheduler: 'Scheduler',
    noScheduler: 'No scheduler data',
    moduleEtags: 'Module Etags',
    noEtags: 'No etags available',
    workspace: 'Workspace',
    noTaskRoot: 'No active task root',
    dimFieldIs: 'is',
  },
  common: {
    dash: '—',
    unknown: 'Unknown',
    attached: 'attached',
    idle: 'idle',
    hundredPercent: '100%',
    newSession: 'New session',
  },
};

export const translations: Record<Lang, Translations> = {
  'zh-CN': zhCN,
  'en-US': enUS,
};
