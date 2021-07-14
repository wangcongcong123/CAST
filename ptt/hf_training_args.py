'''
REFERENCE:

This script is adapted from: https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
'''
import dataclasses
import json
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required
from transformers.trainer_utils import EvaluationStrategy
from transformers.utils import logging

if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)

def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class HFTrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on the command line.

    Parameters:
        output_path (:obj:`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_path (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
            :obj:`output_path` points to a checkpoint directory.
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not.
        do_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation on the dev set or not.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not.
        evaluation_strategy(:obj:`str` or :class:`~transformers.trainer_utils.EvaluationStrategy`, `optional`, defaults to :obj:`"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No evaluation is done during training.
                * :obj:`"steps"`: Evaluation is done (and logged) every :obj:`eval_steps`.
                * :obj:`"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and predictions, only returns the loss.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps: (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            .. warning::

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every ``gradient_accumulation_steps * xxx_step`` training
                examples.
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for Adam.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero).
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            Epsilon for the Adam optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`.
        logging_dir (:obj:`str`, `optional`):
            Tensorboard log directory. Will default to `runs/**CURRENT_DATETIME_HOSTNAME**`.
        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Wheter to log and evalulate the first :obj:`global_step` or not.
        log_and_save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs that equals model saving steps if EvaluationStrategy.STEPS, only log train status if EvaluationStrategy.EPOCH

        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_path`.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 122):
            Random seed for initialization.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details
            on the `apex documentation <https://nvidia.github.io/apex/amp.html>`__.
        local_rank (:obj:`int`, `optional`, defaults to -1):
            During distributed training, the rank of the process.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the mumber of TPU cores (automatically passed by launcher script).
        debug (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When training on TPU, whether to print debug metrics or not.
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`):
            Number of update steps between two evaluations if :obj:`evaluation_strategy="steps"`. Will default to the
            same value as :obj:`log_and_save_steps` if not set.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state and feed it to the model
            at the next training step under the keyword argument ``mems``.
        wandb_run_name (:obj:`str`, `optional`):
            A descriptor for the run. Notably used for wandb logging.
        disable_tqdm (:obj:`bool`, `optional`):
            Whether or not to disable the tqdm progress bars. Will default to :obj:`True` if the logging level is set
            to warn or lower (default), :obj:`False` otherwise.
        remove_unused_columns (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If using `nlp.Dataset` datasets, whether or not to automatically remove the columns unused by the model
            forward method.

            (Note: this behavior is not implemented for :class:`~transformers.TFTrainer` yet.)
        label_names (:obj:`List[str]`, `optional`):
            The list of keys in your dictionary of inputs that correspond to the labels.

            Will eventually default to :obj:`["labels"]` except if the model used is one of the
            :obj:`XxxForQuestionAnswering` in which case it will default to
            :obj:`["start_positions", "end_positions"]`.
    """

    output_path: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    working_dir: str = field(default=".", metadata={"help": "working directory where data are loaded from and models are saved to"})
    overwrite_output_path: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_path points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    # do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_test: bool = field(default=False, metadata={"help": "Whether to do evaluation/predictions on test set."})

    eval_on: str = field(default="acc", metadata={
        "help": "the metric used for patience checking and best ck saving when evaluation during training"})
    patience: int = field(default=20, metadata={"help": "patience, only when evaluation during training is true"})
    evaluate_during_training: bool = field(
        default=None,
        metadata={"help": "Run evaluation during training at each logging step."},
    )
    evaluation_strategy: EvaluationStrategy = field(
        default="no",
        metadata={"help": "Run evaluation during training at each logging step."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                    "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
                    "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    # adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    # adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    # adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps_or_ratio: Union[int, float] = field(default=0, metadata={
        "help": "Linear warmup over warmup steps if > 1 otherwise ratio."})

    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    log_and_save_steps: int = field(default=500, metadata={
        "help": "Log X steps that equals model saving steps if EvaluationStrategy.STEPS"})
    ck_index_select: int = field(default=-1, metadata={
        "help": "0 refers to get the best if exists otherwise the last ck, -1 refers to get the last ck, for selecting ck during testing, only valid when do_test = True"})

    max_seq_length: int = field(default=128, metadata={
        "help": "maximum sequence length for tokenization using BERT-like models, equivalent of max_src_length."})
    max_src_length: int = field(default=128, metadata={
        "help": "maximum source sequence length for tokenization using sequence to sequence models like T5, BART, etc."})
    max_tgt_length: int = field(default=10, metadata={
        "help": "maximum target sequence length for tokenization using sequence to sequence models like T5, BART, etc."})

    # save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_path. Default is unlimited checkpoints"
            )
        },
    )

    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=122, metadata={"help": "random seed for initialization"})

    optimizer: str = field(default="adamw", metadata={
        "help": 'This works when the optimizers argument is None to trainer, default to adamw, abailable: ["adamw", "adam", "adagrad", "adadelta", "sgd"]'})
    scheduler: str = field(default="warmuplinear", metadata={
        "help": 'This works when the optimizers argument is None to trainer, default to warmuplinear, available: ["constantlr", "warmuplinear", "warmupconstant", "warmupcosine","warmupcosinewithhardrestarts"]'})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={"help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"},
    )
    debug: bool = field(default=False, metadata={"help": "Whether to print debug metrics on TPU"})

    use_wandb: bool = field(default=False, metadata={"help": "Whether to use wandb when it is available"})
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging. valid only when use_wandb = True"}
    )
    wandb_project_name: str = field(
        default="huggingface", metadata={"help": "valid only when use_wandb = True"}
    )

    use_comet: bool = field(default=False, metadata={"help": "Whether to use comet when it is available"})
    comet_project_name: str = field(default="huggingface", metadata={"help": "comet_project_name only valid when use_comet = True"})
    comet_api_key: Optional[str] = field(default=None, metadata={"help": "required when use_comet = True"})
    comet_mode: str = field(default="online", metadata={"help": '"online" or "offline"? when use_comet = True'})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    # eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    visible_gpu_devices: Optional[str] = field(
        default=None, metadata={"help": "visible gpu devices"}
    )

    relative_script_dir: Optional[str] = field(
        default=".", metadata={"help": "relative path to customized data and metric scripts with Hf's datasets"}
    )

    def __post_init__(self):
        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
        if self.evaluate_during_training is not None:
            self.evaluation_strategy = (
                EvaluationStrategy.STEPS if self.evaluate_during_training else EvaluationStrategy.NO
            )
            warnings.warn(
                "The `evaluate_during_training` argument is deprecated in favor of `evaluation_strategy` (which has more options)",
                FutureWarning,
            )
        else:
            self.evaluation_strategy = EvaluationStrategy(self.evaluation_strategy)

        # if self.eval_steps is None:
        self.eval_steps = self.log_and_save_steps
        self.keep_ck_num = self.save_total_limit
        self.logging_steps = self.log_and_save_steps
        # if not self.working_dir.startswith(".."):
        self.output_path = os.path.join(self.working_dir, self.output_path)
        if self.use_comet:
            assert self.comet_api_key is not None, "you have to set up comet api key when use_comet = True"

        # if self.use_wandb:
        #     assert self.wandb_run_name is not None, "you have to set up wandb_run_name when use_wandb = True"

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        if self.visible_gpu_devices is not None:
            device_ids = self.visible_gpu_devices.split(',')
            if len(device_ids) <= 0:
                device = torch.device("cpu")
                n_gpu = 0
            else:
                device = torch.device(f"cuda:{device_ids[0]}")
                n_gpu = len(device_ids)
            return device, n_gpu
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        return self._setup_devices[1]

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
