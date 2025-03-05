#!/usr/bin/env python
# coding: utf-8

# In[1]:


import weave
from weave import Dataset as WeaveDataset
from weave import Evaluation as WeaveEvaluation
from openai import OpenAI
import json
import asyncio


# In[2]:


client = OpenAI()


# In[3]:


# Initialise the weave project
weave.init('experiment_weave_dino')


# In[ ]:





# In[4]:


# Weave will track the inputs, outputs and code of this function
@weave.op()
def extract_dinos(sentence: str) -> dict:
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages= [
            {
                "role": "system",
                "content": """In JSON format extract a list of `dinosaurs` with their `name`, their `common_name`, and whether its `diet` is a herbivore or carnivore"""
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# In[ ]:





# In[5]:


@weave.op()
def count_dinos(dino_data: dict) -> int:
    # count the number of items in the returned list
    k = list(dino_data.keys())[0]
    return len(dino_data[k])


# In[6]:


@weave.op()
def dino_tracker(sentence: str) -> dict:
    # extract dinosaurs using a LLM
    dino_data = extract_dinos(sentence)

    # count the number of dinosaurs returned
    dino_data = json.loads(dino_data)
    n_dinos = count_dinos(dino_data)
    return {"n_dinosaurs": n_dinos, "dinosaurs": dino_data}


# In[ ]:





# In[ ]:





# In[7]:


sentence = """I love dinosaurs. In the movie, Tyrannosaurus rex (T. rex) chased after a Triceratops (Trike), both carnivore and herbivore locked in an ancient dance. Meanwhile, a gentle giant Brachiosaurus (Brachi) calmly munched on treetops, blissfully unaware of the chaos below."""


# In[ ]:





# In[8]:


result = dino_tracker(sentence)


# In[9]:


print(f'result = \n{result}')


# In[ ]:





# In[10]:


sentence_2 = "There are no more dinosaurs in this world."


# In[11]:


with weave.attributes({'user_id': 'bikash', 'env': 'development', 'contents': 'not included'}):
    result_2 = dino_tracker(sentence_2)


# In[12]:


print(f'result_2 = \n{result_2}')


# In[ ]:





# In[ ]:





# # Example of a Weave tracked dataset

# In[13]:


# Documentation at https://weave-docs.wandb.ai/guides/core-types/datasets


# In[14]:


example_dataset = WeaveDataset(name='example-dataset', rows=[
    {'id': '0', 'sentence': "He no likes ice cream.", 'correction': "He doesn't like ice cream."},
    {'id': '1', 'sentence': "She goed to the store.", 'correction': "She went to the store."},
    {'id': '2', 'sentence': "They plays video games all day.", 'correction': "They play video games all day."}
])
# Publish the dataset
weave.publish(example_dataset)


# In[15]:


# Retrieve the dataset
dataset_ref = weave.ref('example-dataset').get()


# In[16]:


example_input = dataset_ref.rows[2]['sentence']
example_input


# In[ ]:





# In[ ]:





# # Evaluations

# In[17]:


# Documentation at https://weave-docs.wandb.ai/guides/core-types/evaluations


# In[18]:


# First we will need a dataset. Can use the dataset created above. Alternatively, it can be a list of dictionaries.


# In[19]:


# Then define your custom scoring function.
# Scoring functions need to have a model_output keyword argument, but the other arguments are user defined and are taken from the dataset examples. 
#  It will only take the necessary keys by using a dictionary key based on the argument name.
@weave.op()
def match_score1(correction: str, model_output: dict) -> dict:
    # Here is where you'd define the logic to score the model output
    return {'match': correction == model_output['generated_text']}

# Instantiate an Evaluation object with the specification of the dataset and the scoring function to use for evaluation.
evaluation = WeaveEvaluation(
    dataset=example_dataset, scorers=[match_score1]
)


# In[20]:


# Then we need a model and compute predictions on the dataset using the model.
class MyModel(weave.Model):
    prompt: str

    @weave.op()
    def predict(self, sentence: str):
        # here's where you would add your LLM call and return the output
        return {'generated_text': 'Hello, ' + self.prompt}

model = MyModel(prompt='World')


# In[21]:


# Run evaluation on predictions of the model.
# asyncio.run(evaluation.evaluate(model))
# if you're in a Jupyter Notebook, run:
await evaluation.evaluate(model)


# In[ ]:




