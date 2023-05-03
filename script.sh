#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Usage: ./script.sh <test_dir>"
else
  FILES="${1}/input/*"
  for f in ${FILES}
  do
    b=$(basename -- "$f")
    echo $b
    ./kcliques "$f" 12 temp.txt
    diff temp.txt "${1}/output/$b"
  done
  rm temp.txt
fi
