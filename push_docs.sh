#!/bin/bash

# 设置你的项目路径
PROJECT_DIR="/home/xqgao/2025/MIT/Awesome-Computational-Imaging"
CHAPTERS_DIR="$PROJECT_DIR/chapters"

# 进入项目目录
cd "$PROJECT_DIR" || { echo "❌ Failed to enter $PROJECT_DIR"; exit 1; }

# 确保所有更改保存
git stash -u

# 切换或创建 docs 分支
if git show-ref --verify --quiet refs/heads/docs; then
    git checkout docs
else
    git checkout -b docs
fi

# 清空 docs 分支内容（可选，根据你是否只想保留 chapters）
git rm -rf . > /dev/null 2>&1

# 将 chapters 文件复制过来（保留目录结构）
cp -r "$CHAPTERS_DIR"/* ./

# 添加并提交
git add .
git commit -m "Update chapters content in docs branch"

# 推送到远程
git push -u origin docs

# 回到原来分支
git checkout -

# 恢复 stash
git stash pop

echo "✅ Chapters successfully pushed to docs branch."
