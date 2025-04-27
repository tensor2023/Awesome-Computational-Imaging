cd "$CHAPTERS_DIR"
find . -name "*.ipynb" | while read -r ipynb_file; do
    nb_dir="$(dirname "$ipynb_file")"  # ipynb所在的目录
    jupyter nbconvert --to markdown "$ipynb_file" --output-dir "$nb_dir"
    echo "✅ Converted $ipynb_file -> $nb_dir"
done
echo "🎯 All .ipynb files are now properly converted to .md!"
