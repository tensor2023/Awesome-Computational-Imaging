#!/bin/bash

# 设置路径
SRC_DIR="/home/xqgao/2025/MIT/Awesome-Computational-Imaging"
BOOK_DIR="/home/xqgao/2025/MIT/compimg_book"
CONTENT_DIR="$BOOK_DIR/content"

# 创建 Jupyter Book 项目
jupyter-book create $BOOK_DIR --force
mkdir -p "$CONTENT_DIR"

# 清空 TOC
TOC_FILE="$BOOK_DIR/_toc.yml"
echo "format: jb-book" > $TOC_FILE
echo "root: intro" >> $TOC_FILE
echo "chapters:" >> $TOC_FILE

# 拷贝 intro.md
cp $BOOK_DIR/intro.md $CONTENT_DIR/ 2>/dev/null || echo "# Welcome" > $CONTENT_DIR/intro.md

# 查找所有 ipynb 并拷贝到 content/
find "$SRC_DIR" -name "*.ipynb" | while read file; do
    fname=$(basename "$file")
    chapter_name="${fname%.*}"  # 去除扩展名
    dest="$CONTENT_DIR/$chapter_name.ipynb"
    
    cp "$file" "$dest"
    echo "  - file: content/$chapter_name" >> $TOC_FILE
done

# 构建网页
cd $BOOK_DIR
jupyter-book build .

# 安装 ghp-import（如果没装）
pip install ghp-import

# 发布到 GitHub Pages
ghp-import -n -p -f _build/html

cd ..
echo "✅ 网页构建并已部署到 GitHub Pages！"
