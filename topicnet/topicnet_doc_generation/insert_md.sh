#!/bin/bash
# find all md files

find ../../topicnet -not -path '*/\.[a-z]*/*' -not -path '*documentation*' -not -path '*_doc_generation*' -name README.md -print0 | xargs -0 -I % ./transform_replace.sh % 
