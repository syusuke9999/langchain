#!/bin/bash

# patterns.txt ファイルから改行で区切られたパターンを読み込む
patterns=$(cat secret-patterns.txt)

# 各パターンに対してリポジトリ内をチェック
for pattern in $patterns; do
  echo "Checking for pattern: $pattern"
  git rev-list --all | xargs -L1 git grep -n -E "$pattern"
done
