!/bin/bash
CHAPTERS_DIR = '/home/xqgao/2025/MIT/Awesome-Computational-Imaging/chapters'
cd "$CHAPTERS_DIR"
find . -name "*.ipynb" | while read -r ipynb_file; do
    nb_dir="$(dirname "$ipynb_file")"  # ipynb所在的目录
    jupyter nbconvert --to markdown "$ipynb_file" --output-dir "$nb_dir"
    echo "✅ Converted $ipynb_file -> $nb_dir"
done
echo "🎯 All .ipynb files are now properly converted to .md!"


# cd "$CHAPTERS_DIR"

# # 手动输入要转换的文件名
# read -p "Enter the name of the ipynb file you want to convert (e.g., Chapter01_SIREN.ipynb): " ipynb_file

# # 检查文件是否存在
# if [[ ! -f "$ipynb_file" ]]; then
#     echo "❌ File not found: $ipynb_file"
#     exit 1
# fi

# # 提取目录
# nb_dir="$(dirname "$ipynb_file")"
# base_name="$(basename "$ipynb_file" .ipynb)"

# # 直接用同名.md作为输出
# jupyter nbconvert --to markdown "$ipynb_file" --output "$nb_dir/$base_name"

# echo "✅ Converted $ipynb_file -> $nb_dir/$base_name.md"
