from typing import Dict
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("jsonl_classification_reader")
class JsonlClassificationReader(DatasetReader):
    """
    Reads a file from a classification dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 input: str,
                 gold_label: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._input = input
        self._gold_label = gold_label
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []
        with open(file_path, 'r') as snli_file:
            logger.info("Reading JSONline instances from jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(snli_file):
                example = json.loads(line)
                input = example[self._input]
                label = str(example[self._gold_label])
                instances.append(self.text_to_instance(input, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         input: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        input_tokens = self._tokenizer.tokenize(input)
        fields['tokens'] = TextField(input_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'JsonlClassificationReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        input = params.pop('input', None)
        gold_label = params.pop('gold_label', None)
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return JsonlClassificationReader(tokenizer=tokenizer,
                          token_indexers=token_indexers,
                                         input=input,
                                         gold_label=gold_label)
