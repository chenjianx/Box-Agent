---
name: mcp-config
description: 管理 MCP 服务器配置（mcp.json）：查看、添加、修改、移除、启用和禁用
category: workflow
keywords:
  - mcp
  - mcp配置
  - mcp.json
  - 添加服务器
  - 删除服务器
  - configure mcp
  - add mcp server
---

# MCP 配置管理

`mcp_config` 工具只改 `mcp.json`；写入成功不代表 MCP Server 已连接。宿主会监听
文件变更并自动调 `mcp/reconnect`。只有收到内部 MCP runtime update，才能确认热重连
及工具注册完成；没有宿主时重启 box-agent 才生效。

写入路径优先跟随 loader 当前实际读取的 `mcp.json`；loader 尚未初始化时才回退到
用户目录或开发态配置。

## 操作

```
mcp_config(action="list")
mcp_config(action="inspect_browser")  # 只读返回浏览器模式/Profile 摘要，不暴露其他 MCP 凭据
mcp_config(action="update", name="playwright", config={"args_remove":["--headless"]}) # 有头
mcp_config(action="update", name="playwright", config={"args_add":["--headless"]})    # 无头
mcp_config(action="add", name="my-server", config={"command":"npx","args":["-y","@my/mcp-server"]})
mcp_config(action="add", name="remote", config={"url":"https://example.com/mcp","type":"streamable_http"})
mcp_config(action="enable",  name="my-server")
mcp_config(action="disable", name="my-server")
mcp_config(action="remove",  name="my-server")
```

`update` 只修改提供的字段，其他字段保持不变；还支持 `args_add`、`args_remove` 和
`remove_fields`。`add` 仍用于添加或整条替换配置。

新配置的 MCP 工具默认进入 deferred catalog，不会整套注入模型上下文。若 runtime
update 在当前轮确认注册成功，使用 `tool_search` 搜索并激活需要的真实工具；只有极少数
必须常驻的核心工具才配置 `alwaysLoad: true`。

## 注意

- 启用前先去掉 `disabled` 字段，否则 reconnect 会立刻拒。
- 内置 server 由宿主托管；需要局部调整时使用 `update`，不要用 `add/remove` 整条覆盖。
- 未收到 runtime update 时，只能报告“配置已写入、连接待确认”，不能报告连接成功。
