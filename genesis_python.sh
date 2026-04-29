#!/bin/bash

# 1. 注入我们之前测试成功的无头渲染环境变量
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYGLET_HEADLESS=1

# 2. 定位当前仓库的虚拟环境 Python
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# 3. 直接使用本机独享 GPU 环境运行，并把 VS Code 传来的所有参数 ("$@") 原样转发进去
"$VENV_PYTHON" "$@"
