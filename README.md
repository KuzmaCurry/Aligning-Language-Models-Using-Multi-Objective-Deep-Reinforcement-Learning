**Revised Paper**
================

The revised paper is provided as can be seen in **Revision.pdf**<br>
Revision parts are highlighted in red.
<br><br>

**Dataset**
===
**Reward Models Datasets**
__________________________
RMhelp training dataset is collected from the [PKU-SafeRLHF (helpfulness)](https://github.com/PKU-Alignment/beavertails), [Anthropic Helpful](https://github.com/anthropics/hh-rlhf),
and the [Standford SHP](https://huggingface.co/datasets/stanfordnlp/SHP).<br>
RMsafe training dataset is collected from the [PKU-SafeRLHF (safe)](https://github.com/PKU-Alignment/beavertails) and [Anthropic Harmless](https://github.com/anthropics/hh-rlhf).
<br><br>
**MOLMA Dataset**
_________________
MOLMA training dataset is given in the query_only_MODRL_dataset.json in the MOLMA directory.<br>
The dataset is comprised of unanswered prompts collected from the [Alpaca Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) and the [Anthropic Harmless](https://github.com/anthropics/hh-rlhf) datasets. The text prompts are processed by the [Microsoft Phi-2 tokenizer](https://huggingface.co/microsoft/phi-2).<br> 

**Performance Examples**
=============
We provide generation examples of the MOLMA, the reference model, and the four SODRL models in the performance_examples.json file for a more straightforward performance comparison.<br>
In the performance_example.json is a list of size 100. Each data of the list is a dictionary formatted as:<br>

{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"prompt": ...,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"response1": ...,<br>
&nbsp;&nbsp;&nbsp;&nbsp;"response2": ...,<br>
&nbsp;&nbsp;&nbsp;&nbsp;...<br>
&nbsp;&nbsp;&nbsp;&nbsp;"response6": ...<br>
}<br>

Each dictionary contains one prompt and 6 different responses generated from 6 models based on the prompt. There are 100 dictionaries with 100 different prompts and corresponding responses.<br>
