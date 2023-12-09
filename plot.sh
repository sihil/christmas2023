#!/bin/bash
set -e

FILE=${1}

if [ ! -f "${FILE}" ]; then
    echo "File ${FILE} not found!"
    exit 1
fi

# get the number of lines in the file
LINES=$(wc -l < "${FILE}")

# now plot with a progress bar
gcode-cli/gcode-cli "$FILE" 2>&1 | grep "<< OK" | pv -W -l -s "${LINES}" -pte > /dev/null
