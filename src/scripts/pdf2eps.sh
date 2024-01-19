#!/bin/bash

SOURCE_DIRECTORY="../../images/allin1_viz"
OUTPUT_DIRECTORY="../../images/eps/allin1_viz"

mkdir -p "$OUTPUT_DIRECTORY"

for file in "$SOURCE_DIRECTORY"/*.pdf
do
  base=$(basename "$file" .pdf)

  convert "$file" "$OUTPUT_DIRECTORY/$base.eps"
done

