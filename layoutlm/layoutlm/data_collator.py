import dataclasses
import typing
import transformers


@dataclasses.dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: transformers.LayoutLMv2FeatureExtractor
    tokenizer: transformers.PreTrainedTokenizerBase
    padding: typing.Union[bool, str, transformers.file_utils.PaddingStrategy] = True
    max_length: typing.Optional[int] = None
    pad_to_multiple_of: typing.Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        image = self.feature_extractor([feature["original_image"] for feature in features], return_tensors="pt").pixel_values
        entities = []
        relations = []
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["labels"]
            del feature["original_image"]
            entities.append(feature["entities"])
            del feature["entities"]
            relations.append(feature["relations"])
            del feature["relations"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch["image"] = image
        batch["entities"] = entities
        batch["relations"] = relations
        return batch

