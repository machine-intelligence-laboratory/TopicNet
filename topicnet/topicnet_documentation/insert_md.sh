#!/bin/bash
# find all md files

find ../../topicnet -not -path '*/\.[a-z]*/*' -not -path '*documentation*' -name *.md -print0 | xargs -0 -I % ./transform_replace.sh % 
