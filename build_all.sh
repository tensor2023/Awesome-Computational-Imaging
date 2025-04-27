#!/bin/bash

# === 设置路径 ===
BOOK_DIR="/home/xqgao/2025/MIT/Awesome-Computational-Imaging/compimg_book"
CHAPTERS_DIR="/home/xqgao/2025/MIT/Awesome-Computational-Imaging/chapters"
TOC_FILE="$BOOK_DIR/_toc.yml"
GITHUB_USER="tensor2023"
REPO_NAME="Awesome-Computational-Imaging"

# （可选）如果设置了 chapter 名，只构建这一章
BUILD_CHAPTER=""

# === 0. 处理传入参数（可选分批build）===
if [[ $# -gt 0 ]]; then
    if [[ $1 == --chapter ]]; then
        BUILD_CHAPTER="$2"
        echo "🛠 Only building chapter: $BUILD_CHAPTER"
    fi
fi

# # === 1. ipynb 转为 md（在原目录生成 .md，保留 .ipynb）===
# cd "$CHAPTERS_DIR"
# find . -name "*.ipynb" | while read -r ipynb_file; do
#     nb_dir="$(dirname "$ipynb_file")"  # ipynb所在的目录
#     jupyter nbconvert --to markdown "$ipynb_file" --output-dir "$nb_dir"
#     echo "✅ Converted $ipynb_file -> $nb_dir"
# done
# echo "🎯 All .ipynb files are now properly converted to .md!"


# === 1. 保存当前工作区 ===
echo "💾 Saving current changes..."
git add .
git commit -m "🔖 Save current files before deploy" || echo "⚠️ No changes to commit."

# === 2. 清空 compimg_book/chapters/，重新复制 .md 和 *_files ===
echo "🔄 Preparing compimg_book/chapters..."
rm -rf "$BOOK_DIR/chapters"
mkdir -p "$BOOK_DIR/chapters"
# cd "$CHAPTERS_DIR" TODO

# === 2.1 复制所有 md 文件（保持结构）===
find . -name "*.md" | while read -r md_file; do
    dst_path="$BOOK_DIR/chapters/$(dirname "$md_file")"
    mkdir -p "$dst_path"
    cp "$md_file" "$dst_path/"
done

# === 2.2 复制所有 *_files 文件夹（图片资源）===
find . -type d -name "*_files" | while read -r d; do
    dst_path="$BOOK_DIR/chapters/$(dirname "$d")"
    mkdir -p "$dst_path"
    cp -r "$d" "$dst_path/"
done

echo "✅ All .md and *_files copied to $BOOK_DIR/chapters"

# === 3. 确保 intro.md 存在 ===
if [[ ! -s "$BOOK_DIR/intro.md" ]]; then
    echo "# $REPO_NAME" > "$BOOK_DIR/intro.md"
    echo "✅ Auto-generated intro.md"
fi

# === 4. 检查 TOC 文件是否存在 ===
if [[ ! -f "$TOC_FILE" ]]; then
    echo "❌ Error: _toc.yml not found at $TOC_FILE"
    exit 1
fi

# === 5. 生成 HTML 页面 ===
echo "📘 Building Jupyter Book..."
if [[ -n "$BUILD_CHAPTER" ]]; then
    jupyter-book build "$BOOK_DIR/chapters/$BUILD_CHAPTER"
else
    jupyter-book build "$BOOK_DIR"
fi

if [[ $? -ne 0 ]]; then
    echo "❌ jupyter-book build failed. Please fix your TOC or markdown files first."
    exit 1
fi

# === 6. 切换到 docs 分支 ===
CURRENT_BRANCH=$(git branch --show-current)
echo "🔀 Switching to docs branch..."
git switch docs || { echo "❌ Failed to switch to docs branch."; exit 1; }

# === 7. 清空 docs 分支（保留 .git）===
echo "🧹 Cleaning up docs branch..."
find . -mindepth 1 ! -regex '^\.\/\.git\(/.*\)?' -delete

# === 8. 拷贝 build 出来的 HTML 文件到 docs 分支 ===
echo "📋 Copying built HTML files to docs branch..."
cp -r "$BOOK_DIR/_build/html/"* .

# === 9. 提交并推送到 docs 分支 ===
echo "🚀 Committing and pushing to docs branch..."
git add .
git commit -m "📘 Deploy: Clean HTML build"
git push origin docs --force

# === 10. 切回原分支 master ===
git switch "$CURRENT_BRANCH"

echo ""
echo "✅ Done! Successfully deployed clean HTML to docs branch!"
echo "🔗 View it at: https://$GITHUB_USER.github.io/$REPO_NAME/"
