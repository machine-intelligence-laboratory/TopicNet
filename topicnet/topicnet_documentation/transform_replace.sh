#!/bin/bash
# create and insert htmled markdown 

DIR=$(dirname "$1")
DIR=${DIR#..\/..\/}

pandoc -f markdown -t html $1 -o $DIR/htmled.html 
sed -i '\|</header*|r '$DIR/htmled.html'' "$DIR/index.html"
rm $DIR/htmled.html
sed -i "s/<h1>Index.*/<h1>Документация к модулю <code>topicnet<\/code><\/h1>/" "$DIR/index.html"
