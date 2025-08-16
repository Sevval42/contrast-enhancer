#!/bin/bash
for file in *.vert *.frag *.comp; do
  [ -f "$file" ] || continue
  name="${file%.*}"
  ext="${file##*.}"
  case "$ext" in
    vert) stage="vert" ;;
    frag) stage="frag" ;;
    comp) stage="comp" ;;
    *) echo "Skipping $file"; continue ;;
  esac
  echo "Compiling $file ..."
  glslangValidator -V -S "$stage" "$file" -o "$name.spv"
done