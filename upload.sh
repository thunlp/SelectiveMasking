id=${1:-1}
if [ $id = 1 ]; then
    server="newnlp"
else
    if [ $id = 2 ]; then
        server="newnlp2"
    elif [ $id = 3 ]; then
        server="newnlp3"
    else
        server="nlp193"
    fi
fi
echo ${server}
rsync -avzp ./* ${server}:~/SelectiveMasking
