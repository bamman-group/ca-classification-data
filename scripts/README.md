### Run supervised models (logistic/linear regression, BERT, RoBERTa, Llama 3 8B) on specific GPU device.

```
for task in animacy folktales genre haiku hippocorpus literary_time narrativity emotion soc strangeness
do
./run.sh $task cuda:0
done
```

### Run prompting models with 10 shots (see prompt/ directory for all prompts used in Table 5).

```
# gpt4o via Azure

for task in animacy folktales genre haiku hippocorpus literary_time narrativity emotion soc strangeness
do
python run_azure_chatgpt.py ../data/$task/test.jsonl prompts/$task/${task}_10.txt llm_outputs/gpt4o/$task
done

# llama 3 70B instruct

for task in animacy folktales genre haiku hippocorpus literary_time narrativity emotion soc strangeness
do
python run_llama3.py ../data/$task/test.jsonl prompts/$task/${task}_10.txt llm_outputs/llama3_70b/$task
done

# mixtral 8x22B instruct

for task in animacy folktales genre haiku hippocorpus literary_time narrativity emotion soc strangeness
do
python run_mixtral.py ../data/$task/test.jsonl prompts/$task/${task}_10.txt llm_outputs/mixtral_8x22b/$task
done

```

### Use GPT4o for category sensemaking

```
mkdir gpt_sensemaking

# classification

for task in animacy folktales genre haiku hippocorpus narrativity emotion soc strangeness
do
python gpt_category_understanding.py ../data/$task/train.jsonl gpt_sensemaking/$task.out.txt
done

# regression

python gpt_regression_understanding.py ../data/literary_time/train.jsonl gpt_sensemaking/literary_time.out.txt


```


