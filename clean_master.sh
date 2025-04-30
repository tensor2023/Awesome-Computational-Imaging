# 1. 在本地新建一个临时分支（基于当前本地 README）
git checkout --orphan clean_master

# 2. 删除所有文件，只保留 README.md
find . -not -name 'README.md' -not -path './.git/*' -type f -delete
find . -type d -empty -delete

# 3. 添加并提交 README.md
git add README.md
git commit -m "Reset master with clean README.md only"

# 4. 强推覆盖远程 master（⚠️慎用）
git push -f origin clean_master:master

# 5. 回到你本地原来的分支，继续正常开发
git checkout master
