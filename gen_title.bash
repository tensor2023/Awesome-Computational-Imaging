cd /home/xqgao/2025/MIT/Awesome-Computational-Imaging/compimg_book/chapters

find . -name "*.md" | while read -r md; do
    first_line=$(head -n 1 "$md")
    if [[ ! $first_line =~ ^# ]]; then
        title="# $(basename "$md" .md)"
        sed -i "1i $title\n" "$md"
        echo "✔ Added title to $md"
    fi
done
