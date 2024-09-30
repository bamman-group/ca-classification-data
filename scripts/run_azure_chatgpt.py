from openai import AzureOpenAI
import os, sys, json

api_key = # put api_key here
deployment_name=# put deployment name here
endpoint=# end point here

model=# model name here


client = AzureOpenAI(
	api_key=api_key, 
	api_version="2024-02-01",
	azure_endpoint = endpoint
)

def run_prompt(prompt):
	completion = client.completions.create(

		prompt=prompt,

		model=deployment_name,
		temperature=0.0
	)
	return completion


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

			text=' '.join(text.split(" ")[:1])

			if not os.path.exists(outfile):

				with open(outfile, "w") as out:

					prompt=get_prompt(text, template)
					completion=run_prompt(prompt)
					json_string=completion.model_dump_json()
					print(json_string)
					out.write("%s\t%s\t%s\n" % (idx, label, json_string))

infile=sys.argv[1]
prompt_file=sys.argv[2]
outfolder=sys.argv[3]

# args: 
	# input jsonl file (with text + labels)
	# prompt file
	# output directory to store API output (one file per request)

# python haiku/test.jsonl prompt_file.txt haiku_out

proc_file(infile, prompt_file, outfolder)

