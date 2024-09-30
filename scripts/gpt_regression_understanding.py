import os, sys, json
from openai import OpenAI
from random import shuffle
import numpy as np
import tiktoken

client = OpenAI(
	api_key=os.environ.get("OPENAI_API_KEY"),
)

def run_prompt(prompt):
	completion = client.chat.completions.create(
		messages=[
			{
				"role": "user",
				"content": prompt,
			}
		],
		model="gpt-4o",
		temperature=0.0
	)
	return completion


def get_prompt(data):

	prompt="""Consider the data below, which contains a list of text/label pairs that illustrate a regression problem:

	%s

	Using this data, and this data alone, what are the textual characteristics that differentiate the two ends of this scale (let us call those ends "Pole 1" and the "Pole 2")?  The label "-1" is closer to Pole 1 than the label "1" is; the label "1", likewise, is closer to Pole 2 than the label "-1" is.  Provide a list of bullet points of those textual features in the following format:

	### Pole 1 Characteristics:

	- **
	- **


	...

	### Pole 2 Characteristics:

	- **
	- **


	""" % (json.dumps(data))
	
	return prompt



def proc_file(filename, output_file):


	enc = tiktoken.encoding_for_model("gpt-4o")
	max_tokens=100000
	cur_tokens=0

	with open(output_file,"w") as out:
		data_points=[]

		with open(filename) as file:
			for idx, line in enumerate(file):
				data=json.loads(line.rstrip())
				cat=float(data["label"])
				text=data["text"]

				tokens = enc.encode(json.dumps({"text": text, "label": cat}))
				cur_tokens+=len(tokens)

				print(cur_tokens)

				if cur_tokens >= max_tokens:
					break



				data_points.append({"text": text, "label": cat})

		shuffle(data_points)

		

		prompt=get_prompt(data_points)

		total_toks=enc.encode(prompt)
		print("total_toks", len(total_toks))


		completion=run_prompt(prompt)
		json_string=completion.model_dump_json()
		print(json_string)
		out.write("%s\t%s\t%s\n" % ("", json.dumps(prompt[:500]), json_string))

infile=sys.argv[1]
outfile=sys.argv[2]

# args: 
	# input jsonl file (with text + labels)
	# output file to store API output

# python haiku/train.jsonl output.txt

proc_file(infile, outfile)

