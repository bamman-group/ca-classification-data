import transformers, os, sys, json
import torch

os.environ['HF_TOKEN']=# put Huggingface access token here

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def run_prompt(prompt):

    messages = [
        {"role": "system", "content": "You are a helpful assistant; follow the instructions in the prompt."},
        {"role": "user", "content": prompt},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=0.9,
    )

    return outputs


def get_prompt(text, template):
    prompt="""%s

Input: %s

    """ % (template, text)
    
    return prompt



def proc_file(filename, prompt_file, output_folder):
    with open(prompt_file) as file:
        template=file.read()

    os.makedirs(output_folder, exist_ok=True)
    with open(filename) as file:
        for idx, line in enumerate(file):
            data=json.loads(line.rstrip())
            label=data["label"]
            text=data["text"]
            outfile="%s/%s.txt"  % (output_folder, idx)
            
            if not os.path.exists(outfile):

                with open(outfile, "w") as out:

                    prompt=get_prompt(text, template)
                    completion=run_prompt(prompt)
                    print(completion)
                    out.write("%s\t%s\t%s\n" % (idx, label, json.dumps(completion)))

infile=sys.argv[1]
prompt_file=sys.argv[2]
outfolder=sys.argv[3]

# args: 
    # input jsonl file (with text + labels)
    # prompt file
    # output directory to store API output (one file per request)

# python haiku/test.jsonl prompt_file.txt haiku_out

proc_file(infile, prompt_file, outfolder)




