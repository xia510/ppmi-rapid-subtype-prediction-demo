# Windows 一键启动设计

## 目标

让 Windows 用户只需双击一个批处理文件，即可启动 FastAPI 与 Streamlit。首次启动时，脚本在本机提示输入 DeepSeek API Key，并将它保存到被 Git 忽略的 `.env`；后续启动自动读取，不再重复输入。

## 组件与数据流

- `start_demo.bat`：双击入口，仅调用 PowerShell 启动脚本。
- `scripts/start_demo.ps1`：读取受限的 `.env` 键、寻找 Python、启动两个服务、等待健康检查并打开网页。
- `stop_demo.bat`：双击入口，仅调用 PowerShell 停止脚本。
- `scripts/stop_demo.ps1`：读取本地 PID 文件，并在确认命令行属于本项目后停止进程。
- `.runtime/`：保存 PID 和 stdout/stderr 日志，不提交 Git。
- `.env`：保存本机密钥与可选 Python 路径，不提交 Git。

启动脚本不会使用 dot-source、`Invoke-Expression` 或动态执行 `.env` 内容，只解析允许列表中的 `KEY=VALUE`。密钥不输出到控制台或日志。若 `.env` 不存在，脚本从 `.env.example` 创建；若密钥为空，使用安全输入提示并更新本地文件。

## 失败处理与验证

若端口已被未受管理的进程占用，脚本停止并给出明确提示，不会任意结束其他程序。FastAPI 健康检查失败时，不继续打开网页，并提示查看 `.runtime` 日志。停止脚本只处理 PID 文件记录且命令行符合预期的 Python 进程。

自动化测试覆盖 `.env` 解析、密钥不回显、空密钥报错、恶意 `.env` 内容不执行，以及启动脚本的只校验模式。完整测试套件在交付前运行。
