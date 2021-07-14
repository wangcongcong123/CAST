# coding=utf-8
# This script is finished following HF's datasets' template:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
# More examples as references to write a customized dataset can be found here:
# https://github.com/huggingface/datasets/tree/master/datasets

from __future__ import absolute_import, division, print_function
import json
import os
import datasets

_CITATION = """\

"""
_DESCRIPTION = """\
"""

_EXAMPLES_DOWNLOAD_URL = "data/crisis_t6/examples.json"

class CrisisT6Config(datasets.BuilderConfig):
    def __init__(
            self,
            **kwargs,
    ):
        # self.second_choice=kwargs.pop("second_choice",None)
        super(CrisisT6Config, self).__init__(version=datasets.Version("0.0.0", ""), **kwargs)

class CrisisT6(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CrisisT6Config(
            name="normal",
            description="text as the source and sentiment label as the target",
        ),
        CrisisT6Config(
            name="normal_combine",
            description="all examples instead of splits",
        ),
        CrisisT6Config(
            name="t2t",
            description="target is the sentiment label and source is constructed by QA template: context: ... question: ",
        ),
        CrisisT6Config(
            name="t2t_combine",
            description="all examples instead of splits",
        ),
        CrisisT6Config(
            name="t2t_no_loc_combine",
            description="all examples instead of splits",
        ), CrisisT6Config(
            name="t2t_sep_loc_combine",
            description="all examples instead of splits",
        ),
        CrisisT6Config(
            name="t2t_simple",
            description="all examples instead of splits",
        )
    ]
    """customize dataset."""
    # VERSION = datasets.Version("0.0.0")
    def _info(self):
        data_info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "event_name": datasets.Value("string"),
                    "set_name": datasets.Value("string"),
                    "question_template":datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="#",
            citation=_CITATION,
        )
        return data_info

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_EXAMPLES_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "normal" or self.config.name == "normal_combine":
                    yield id_, {
                        "event_name":data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": data["text"],
                        "target": data["target"],
                    }
                elif self.config.name == "t2t" or self.config.name == "t2t_combine":
                    yield id_, {
                        "event_name":data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": data["source"],
                        "target": data["target"],
                    }
                elif self.config.name == "t2t_no_loc_combine":
                    yield id_, {
                        "event_name":data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": "content: "+data["text"]+f" question: Is this message relevant to {data['event_name'].split()[-1]}?",
                        "target": data["target"],
                    }
                elif self.config.name == "t2t_simple":
                    yield id_, {
                        "event_name": data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": "content: " + data["text"] + f" question: {data['event_name']}?",
                        "target": data["target"],
                    }
                elif self.config.name == "t2t_sep_loc_combine":
                    yield id_, {
                        "event_name": data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": "content: "+data["text"]+f" question: Is this message relevant to a {data['event_name'].split()[-1]} event that occurred in {data['event_name'].split()[0]}?",
                        "target": data["target"],
                    }
