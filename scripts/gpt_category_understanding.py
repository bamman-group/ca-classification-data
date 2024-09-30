import os, sys, json
from openai import OpenAI
from random import shuffle
import numpy as np
import tiktoken

client = OpenAI(
	# This is the default and can be omitted
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


def get_prompt(mapper, data):
	cats=np.arange(0,len(mapper)).tolist()
	# cats=sorted(cats)
	prompt="""Consider the data below, which contains a list of text/label pairs that illustrate a set of categories (%s):

	%s

	Using this data, and this data alone, what are the textual characteristics that differentiate the classes from each other?  Provide a list of bullet points of those textual features in the following format:

	### Category 0 Characteristics:

	- **
	- **


	...

	### Category %s Characteristics:

	- **
	- **


	""" % (', '.join([str(x) for x in cats]), json.dumps(data), cats[-1])
	
	return prompt



def proc_file(filename, output_file):


	mapper={}
	enc = tiktoken.encoding_for_model("gpt-4o")
	max_tokens=100000
	cur_tokens=0



	with open(output_file,"w") as out:
		data_points=[]

		with open(filename) as file:
			for idx, line in enumerate(file):
				data=json.loads(line.rstrip())
				label=data["label"]
				text=data["text"]

				if label not in mapper:
					mapper[label]=len(mapper)


				cat=mapper[label]

				tokens = enc.encode(json.dumps({"text": text, "label": cat}))
				cur_tokens+=len(tokens)

				print(cur_tokens)

				if cur_tokens >= max_tokens:
					break



				data_points.append({"text": text, "label": cat})

		shuffle(data_points)

		

		prompt=get_prompt(mapper, data_points)

		total_toks=enc.encode(prompt)
		print("total_toks", len(total_toks))


		completion=run_prompt(prompt)
		json_string=completion.model_dump_json()
		print(json_string)
		out.write("%s\t%s\t%s\n" % (json.dumps(mapper), json.dumps(prompt[:500]), json_string))

infile=sys.argv[1]
outfile=sys.argv[2]

# args: 
	# input jsonl file (with text + labels)
	# prompt file
	# output directory to store API output (one file per request)

# python haiku/test.jsonl prompt_file.txt haiku_out

proc_file(infile, outfile)

