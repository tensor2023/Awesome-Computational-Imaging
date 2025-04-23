#!/bin/bash

ROOT="/home/xqgao/2025/MIT/Awesome-Computational-Imaging"
GITIGNORE="$ROOT/.gitignore"

# 查找大于 1MB 且不是 .py/.ipynb/.yaml 的文件
find "$ROOT" -type f -size +500k \
    ! -name "*.py" \
    ! -name "*.ipynb" \
    ! -name "*.yaml" \
    -exec du -h {} + |
    sort -hr |
    awk -v root="$ROOT" '{sub(root"/", "", $2); print $2}' >> "$GITIGNORE"

echo "✅ 已将大于 1MB 且非代码文件的路径追加到 $GITIGNORE"
