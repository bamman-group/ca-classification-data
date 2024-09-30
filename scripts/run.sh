folder=$1
device=$2

task="classification"
metric="accuracy"

if [ $folder == "literary_time" ]; then
	task="regression"
	metric="rho"
fi

mkdir -p ../logs/$folder/

python linear_optim.py --input ../data/$folder --task $task --metric $metric > ../logs/$folder/linear.log 2>&1
python bert_optim.py --input ../data/$folder --task $task --metric $metric --device $device > ../logs/$folder/bert.log 2>&1
python roberta_optim.py --input ../data/$folder --task $task --metric $metric --device $device > ../logs/$folder/roberta.log 2>&1
python llama_optim.py --input ../data/$folder --task $task --metric $metric --device $device > ../logs/$folder/llama.log 2>&1
