#!/usr/bin/env python
# coding: utf-8

# In[1]:


import weave
from weave import Dataset as WeaveDataset
from openai import OpenAI
import json


# In[2]:


client = OpenAI()


# In[3]:


# Initialise the weave project
weave.init('experiment_weave_dino')


# In[ ]:





# In[17]:


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





# In[18]:


@weave.op()
def count_dinos(dino_data: dict) -> int:
    # count the number of items in the returned list
    k = list(dino_data.keys())[0]
    return len(dino_data[k])


# In[19]:


@weave.op()
def dino_tracker(sentence: str) -> dict:
    # extract dinosaurs using a LLM
    dino_data = extract_dinos(sentence)

    # count the number of dinosaurs returned
    dino_data = json.loads(dino_data)
    n_dinos = count_dinos(dino_data)
    return {"n_dinosaurs": n_dinos, "dinosaurs": dino_data}


# In[ ]:





# In[20]:





# In[21]:


sentence = """I love dinosaurs. In the movie, Tyrannosaurus rex (T. rex) chased after a Triceratops (Trike), both carnivore and herbivore locked in an ancient dance. Meanwhile, a gentle giant Brachiosaurus (Brachi) calmly munched on treetops, blissfully unaware of the chaos below."""


# In[ ]:





# In[22]:


result = dino_tracker(sentence)


# In[23]:


print(f'result = \n{result}')


# In[ ]:





# In[24]:


sentence_2 = "There are no more dinosaurs in this world."


# In[25]:


with weave.attributes({'user_id': 'bikash', 'env': 'development', 'contents': 'not included'}):
    result_2 = dino_tracker(sentence_2)


# In[26]:


print(f'result_2 = \n{result_2}')


# In[ ]:





# In[ ]:





# # Example of a Weave tracked dataset

# In[29]:


example_dataset = WeaveDataset(name='example-dataset', rows=[
    {'id': '0', 'sentence': "He no likes ice cream.", 'correction': "He doesn't like ice cream."},
    {'id': '1', 'sentence': "She goed to the store.", 'correction': "She went to the store."},
    {'id': '2', 'sentence': "They plays video games all day.", 'correction': "They play video games all day."}
])
# Publish the dataset
weave.publish(example_dataset)


# In[30]:


# Retrieve the dataset
dataset_ref = weave.ref('example-dataset').get()


# In[31]:


example_input = dataset_ref.rows[2]['sentence']
example_input


# In[ ]:





# In[ ]:




