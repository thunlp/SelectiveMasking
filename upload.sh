id=${1:-1}
if [ $id = 1 ]; then
    server="newnlp"
else
    if [ $id = 2 ]; then
        server="newnlp2"
    else
        server="v100"
    fi
fi
echo ${server}
rsync -avzp ./* ${server}:~/nvidia-bert
