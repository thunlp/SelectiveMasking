id=${1:-1}
if [ $id = 1 ]; then
    server="newnlp"
else
    server="newnlp2"
fi

rsync -avzp ./* ${server}:~/nvidia-bert
