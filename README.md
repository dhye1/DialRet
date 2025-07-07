## DialRet: Enhancing Dialogue Retention for Multi-session Conversations


<p align="center">
<picture>
    <img src="./main_fig.png" width="100%" style="margin: 0px auto;">
</picture>

Proceedings of PAKDD 2025 paper "DialRet: Enhancing Dialogue Retention for Multi-Session Conversations" 
Yohan Na*, Dahye Kim*, and Dong-Kyu chae.

<p align="center"> 🤗 <a href="https://huggingface.co/datasets/DILAB-HYU/MSC_bench">Datasets</a> &nbsp | &nbsp 📜 <a href="https://">Paper</a> | &nbsp 💻 <a href="https://github.com/"> Github </a>

> [!Note]
> The paper is written from a multi-session dialogue perspective, which is far from the instruction performance targeted by recent models. 

## Table of Contents

- [Introduction](#introduction)
- [Model Performance](#performance)
- [Quickstart](#quickstart)
- [License](#license)
- [Citation](#citation)
- [Contributors](#contributors)
- [Contact](#contact)

<br>

## Introduction
DialRet is a dialogue-specific language model designed for multi-session conversations. 
Instead of using memory modules, it leverages long-context LMs and instruction-tuning across eight tasks (e.g., dialogue generation, summarization, speaker relation extraction). 
It improves understanding and retention of past dialogues.

The paper also introduces MSC-Bench, a benchmark evaluating dialogue models on memorability, specificity, engagement, and humanness. 
Experiments show DialRet outperforms existing models in multi-session dialogue quality and retention.

### Model Performance
Below are partial report on the performance of the `DialRet`. Please refer to the [Paper](https://) for the full results.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6152b4b9ecf3ca6ab820e325/RNvRrtb8UdZRP3xwikyD-.png)



## Quickstart

#### Example Usage for `DialRet`
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "DILAB-HYU/DialRet-L1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

session_num = 3
session_role_1 = "Neighbors A"
session_role_2 = "Neighbors B"
session_system_prompt = f"You will be shown a {session_num} session dialogues between {session_role_1} and {session_role_2}. Please read and understand given multiple Dialogue Session, then complete the task under the guidance of Task Introduction.\n"

session_input = """\n```
Dialogue Session #1:
Neighbors A:Hi there! I saw your cat in my backyard earlier. She's quite cute. What's her name?
Neighbors B:Oh, thanks! Her name is Luna. She's a rescue cat.
Neighbors A:That's really cool. How old is she?
Neighbors B:She's about 2 years old.
Neighbors A:Does she like being outside?
Neighbors B:Not really. She's pretty much an indoor cat. She likes to snuggle up and sleep all day.
Neighbors A:That's adorable. My kids would love her!
Neighbors B:You're welcome to come over and visit her anytime.
Neighbors A:Thanks, I'd love to! By the way, did you get your fence fixed?
Neighbors B:Yes, we had it repaired last weekend. It was a relief to finally get it fixed. 
Neighbors A:I'm glad to hear that. Did you have to call in a professional?
Neighbors B:Yeah, we had to call a fencing company to come and take care of it. They did a great job though, so we're happy with the results.
Neighbors A:Good to know! I may have to call them too if I ever need fence repairs.
Neighbors B:Absolutely, I can give you their contact information if you'd like.
Neighbors A:Thanks, I appreciate it. Anyway, I won't keep you too long. Thanks for telling me about Luna!
Neighbors B:No problem, happy to talk about her. See you later!
```\n
\n```
Dialogue Session #2:
Neighbors A:Can you believe it? A tree just fell on my car!
Neighbors B:Oh no! Are you okay?
Neighbors A:Yeah, luckily I wasn't in it at the time. But my car is completely totaled.
Neighbors B:That's terrible. Did you call your insurance company?
Neighbors A:Not yet, I'm still in shock. Plus, I was in the middle of reading a really interesting book about philosophy.
Neighbors B:Oh, what book are you reading?
Neighbors A:It's called "The Republic" by Plato. It's all about the concept of justice and government.
Neighbors B:That sounds really fascinating. I've always been interested in philosophy, but I never know where to start.
Neighbors A:Well, "The Republic" is a classic. But if you're just starting out, I'd recommend "Meditations" by Marcus Aurelius. It's a great introduction to Stoicism.
Neighbors B:Thanks for the recommendation. I'll definitely check it out. But in the meantime, let's get your car situation sorted out. Do you need any help with anything?
Neighbors A:That would be great, actually. Do you have any experience dealing with insurance companies?
```\n
\n``` 
Dialogue Session #3:
Neighbors A:Hey, Neighbors B. I have a bit of a problem and was hoping you could help me out.
Neighbors B:Sure thing! What's going on?
Neighbors A:Well, I'm having some trouble with my computer. It's just not working the way it should be, and I don't know what to do.
Neighbors B:Ah, I see. What kind of issues are you having?
Neighbors A:The screen keeps freezing up, and I can't seem to get anything done. I'm really getting frustrated because I have some important work that needs to be done.
Neighbors B:Hmm, that sounds really frustrating. I think I might be able to help, though. Have you tried restarting your computer?
Neighbors A: ###
```\n"""

session_task = """```Task Introduction:
After reading the entire Dialogue Sessions, please create an appropriate response.
```\n
Task Result:"""

input_text= session_system_prompt + session_input + session_task
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(inputs, max_new_tokens=4096, do_sample=False) # Finetuned with length 8192
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Output:
# Neighbors A:Yeah, I've been trying that but it doesn't seem to be helping.
```

<br>

## License
The `DialRet` models are licensed under [MIT License](https://opensource.org/license/mit).
<br>
 
## Citation
```
@article{2025dialret,
      title={DialRet: Enhancing Dialogue Retention forMulti-session Conversations}, 
      author={Yohan Na, Dahye Kim, Dong-kyu Chae},
      year={2025},
      url={}, 
}
```

<br>
