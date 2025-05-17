#!/bin/bash

# 自动更新脚本：同步 upstream/main 到你的主分支并合并到开发分支
# 用法：bash sync_upstream.sh your-dev-branch  (默认分支为 cody)

set -e

# 设置默认开发分支
DEV_BRANCH=${1:-cody}

echo "🔄 开始同步 upstream/main..."

# 确保在 main 分支
git checkout main

# 获取 upstream 最新代码
git fetch upstream

# 合并 upstream/main 到本地 main
git merge upstream/main

# 推送更新后的 main 到你自己的仓库
git push origin main

echo "✅ 已同步 upstream/main 并推送到 origin/main"

# 切换到你的开发分支
git checkout "$DEV_BRANCH"

# 合并 main 分支的内容到你的开发分支
git merge main

echo "✅ 已将 main 合并到 $DEV_BRANCH 分支"

# 可选：推送你的开发分支更新
git push origin "$DEV_BRANCH"

echo "🚀 所有操作完成！现在 $DEV_BRANCH 包含最新的 upstream/main 内容。"
