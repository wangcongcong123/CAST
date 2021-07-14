## Source code for CAST: Crisis Domain Adaptation UsingSequence-to-sequenceTransformers ([Paper](https://www.lill.is/site/pubs/Wang2021a.html), [BibTeX](https://www.lill.is/site/bibtex/Wang2021a.bibtex.txt), Accepted to ISCRAM 2021, CorePaper)

### Quick start

Download the code

```bash
git clone https://github.com/wangcongcong123/CAST.git
cd CAST
```
Download the dataset from [here](https://drive.google.com/file/d/1Rz15fOouD4jTOlkCl_YqDVyAfa9VIW7-/view?usp=sharing), and extract the data to `data/` dir (create it first if not exists).

Model training and testing for crisis domain adaptation:

```bash
# go to your python env
# install dependencies
pip install -r requirements.txt

# training and testing at one go

# here we run CAST on crisis_t6 as an example

python train_t6.py

# In train_t6.py, for quick configuration:

train_event_names => the source event(s)
test_event_name => the target event
data_config => postfix template: 't2t' (postQ) or 'normal' (standard) as described in the paper
model_select => the base seq2seq model: 't5-small' or 't5-base'

# For other configuration, just go for a bit hacking so should be easy.

# For nepal_queensland, similary run `python train_nepal_queensland.py`, go check and configure the script to reproduce the paper's results.

```

### Cite

If you find this helpful for your work, consider to cite it as follows please:
```
@inproceedings{Wang2021a,
 title = {Crisis {{Domain Adaptation Using Sequence}}-to-Sequence {{Transformers}}},
 booktitle = {{{ISCRAM}} 2021 {{Conference Proceedings}} - 18th {{International Conference}} on {{Information Systems}} for {{Crisis Response}} and {{Management}}},
 author = {Wang, Congcong and Nulty, Paul and Lillis, David},
 year = {2021},
 month = {May},
 address = {{Blacksburg, VA, USA}},
 abstract = {User-generated content (UGC) on social media can act as a key source of information for emergency responders in crisis situations. However, due to the volume concerned, computational techniques are needed to effectively filter and prioritise this content as it arises during emerging events. In the literature, these techniques are trained using annotated content from previous crises. In this paper, we investigate how this prior knowledge can be best leveraged for new crises by examining the extent to which crisis events of a similar type are more suitable for adaptation to new events (cross-domain adaptation). Given the recent successes of transformers in various language processing tasks, we propose CAST: an approach for Crisis domain Adaptation leveraging Sequence-to-sequence Transformers. We evaluate CAST using two major crisis-related message classification datasets. Our experiments show that our CAST-based best run without using any target data achieves the state of the art performance in both in-domain and cross-domain contexts. Moreover, CAST is particularly effective in one-to-one cross-domain adaptation when trained with a larger language model. In many-to-one adaptation where multiple crises are jointly used as the source domain, CAST further improves its performance. In addition, we find that more similar events are more likely to bring better adaptation performance whereas fine-tuning using dissimilar events does not help for adaptation. To aid reproducibility, we open source our code to the community.},
}
```
