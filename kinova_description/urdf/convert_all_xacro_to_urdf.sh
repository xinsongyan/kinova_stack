#!/bin/bash
# Converts all .xacro files in the current directory and subdirectories to .urdf files

find . -name "*.xacro" | while read -r xacro_file; do
    urdf_file="${xacro_file%.xacro}.urdf"
    echo "Converting $xacro_file -> $urdf_file"
    xacro "$xacro_file" -o "$urdf_file"
done