#!/bin/bash

# === 配置 ===
BOOK_DIR="/home/xqgao/2025/MIT/Awesome-Computational-Imaging/compimg_book"
GITHUB_USER="tensor2023"
REPO_NAME="Awesome-Computational-Imaging"

# === 1. 保存 master 上的修改 ===
echo "💾 Saving changes on master branch..."
git add .
git commit -m "🔖 Save: Update source files before deploying" || echo "⚠️ No changes to commit."

# 记录当前分支名
CURRENT_BRANCH=$(git branch --show-current)

# === 2. 切换到 docs 分支 ===
echo "🔀 Switching to docs branch..."
git switch docs || { echo "❌ Failed to switch to docs branch."; exit 1; }

# === 3. 清空 docs 分支（保留 .git 目录）===
echo "🧹 Cleaning up docs branch..."
find . -mindepth 1 ! -regex '^\.\/\.git\(/.*\)?' -delete

# === 4. 拷贝 build 出来的 HTML 文件到 docs 分支 ===
echo "📋 Copying built HTML files to docs branch..."
cp -r "$BOOK_DIR/_build/html/"* .

# === 5. 提交清空+网页内容到 docs 分支 ===
echo "🚀 Committing changes to docs branch..."
git add .
git commit -m "📘 Deploy: Clean rebuilt HTML site"

# === 6. 强制推送到远端 docs 分支，覆盖远端旧内容 ===
echo "🚀 Pushing to GitHub (force update)..."
git push origin docs --force

# === 7. 切回原分支 master ===
git switch "$CURRENT_BRANCH"

echo ""
echo "✅ Deployment complete! Your clean website is now live!"
echo "🔗 View it at: https://$GITHUB_USER.github.io/$REPO_NAME/"
