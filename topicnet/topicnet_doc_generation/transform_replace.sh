#!/bin/bash
# create and insert htmled markdown 

DIR=$(dirname "$1")
DIR=${DIR#..\/..\/}

echo $DIR

pandoc -f markdown -t html $1 -o $DIR/htmled.html 
# echo "created html-file from markdown $1 in $DIR"
# ls $DIR
sed -i '\|</header*|r '$DIR/htmled.html'' "$DIR/index.html"

rm $DIR/htmled.html

# echo "removed html-file and created index.html"
# ls $DIR
# echo ""
sed -i "s/<h1>Index.*/<h1><code>TopicNet<\/code> library documentation <\/h1>/" "$DIR/index.html"
