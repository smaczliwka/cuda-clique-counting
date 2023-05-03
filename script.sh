#!/bin/bash
if [ $# -ne 2 ]; then
  echo "Usage: ./script.sh <test_dir> <k>"
else
  FILES="${1}/input/*"
  for f in ${FILES}
  do
    b=$(basename -- "$f")
    echo $b
    ./kcliques "$f" "${2}" temp.txt
    diff temp.txt "${1}/output/$b"
  done
  rm temp.txt
fi
