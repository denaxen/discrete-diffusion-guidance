import functools
import itertools
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import numpy as np
import tokenizers
import torch
import transformers

import custom_datasets.discretized_cifar10
import custom_datasets.ten_species_dataset
import utils

LOGGER = utils.get_logger(__name__)


# noinspection RegExpRedundantEscape
def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90_000_000],
        'validation': rawdata[90_000_000: 95_000_000],
        'test': rawdata[95_000_000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos,
                 add_special_tokens=True):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  # `-2` to account for [BOS] and [EOS] to be added below
  new_block_size = block_size - (2 if add_special_tokens else 0)
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    if add_special_tokens:
      _values.append(
        [bos]
        + concatenated_examples[i : i + new_block_size]
        + [eos])
    else:
      _values.append(
        concatenated_examples[i: i + new_block_size])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)),
    streaming=False, override_cache=False,
    add_special_tokens=True,
    label_col=None, label_threshold=None):
  if label_col is not None:
    label_suffix = f'_label-{label_col}'
    if label_threshold is not None:
      label_suffix += f'_threshold-{label_threshold}'
  else:
    label_suffix = ''
    
  # Special filename for lm1b validation split to avoid conflicts
  if dataset_name == 'lm1b' and mode == 'validation':
    dataset_name_for_cache = 'lm1b-val'
  else:
    dataset_name_for_cache = dataset_name
    
  if wrap:
    filename = f'{dataset_name_for_cache}_{mode}_bs{block_size}_wrapped{label_suffix}.dat'
  else:
    filename = f'{dataset_name_for_cache}_{mode}_bs{block_size}_unwrapped{label_suffix}.dat'
  _path = os.path.join(cache_dir, filename)
  if utils.fsspec_exists(_path) and not override_cache:
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for subsampling
    block_size *= 2

  if dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'amazon_polarity':
    dataset = datasets.load_dataset(
      'amazon_polarity',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'qm9':
    dataset = datasets.load_dataset(
      'yairschiff/qm9',
      cache_dir=cache_dir,
      streaming=streaming,
      split='train')  # Dataset only has 'train' split
    if label_threshold is not None:
      pctiles = label_threshold if isinstance(label_threshold, list) \
        else [label_threshold]
      pctile_values = np.percentile(dataset[label_col],
                                    q=pctiles)
      threshold = np.ones(len(dataset[label_col])) * len(pctiles)
      for i, p in reversed(list(enumerate(sorted(pctile_values)))):
        threshold[dataset[label_col] <= p] = i
      dataset = dataset.add_column(
        f"{label_col}_threshold", threshold.astype(int))
      label_col = f"{label_col}_threshold"
    dataset = dataset.train_test_split(
      test_size=0.05, seed=42)  # hard-coded seed & size
    dataset = dataset[mode]
  elif dataset_name == 'ten_species':
    return custom_datasets.ten_species_dataset.TenSpeciesDataset(
      split=mode,
      tokenizer=tokenizer,
      max_length=block_size,
      rc_aug=False,  # TODO: find way to pass this
      add_special_tokens=add_special_tokens)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  # Special handling for lm1b: create validation split from train split
  if dataset_name == 'lm1b' and mode in ['train', 'validation']:
    # Check if we already have the splits cached
    lm1b_train_cache = os.path.join(cache_dir, 'lm1b_train_split.dat')
    lm1b_val_cache = os.path.join(cache_dir, 'lm1b_val_split.dat')
    
    if (utils.fsspec_exists(lm1b_train_cache) and 
        utils.fsspec_exists(lm1b_val_cache) and 
        not override_cache):
      # Load cached splits
      if mode == 'train':
        data = datasets.load_from_disk(lm1b_train_cache)
      else:  # mode == 'validation'
        data = datasets.load_from_disk(lm1b_val_cache)
    else:
      # Create the splits
      LOGGER.info('Creating lm1b train/validation split (300K samples for validation)')
      full_train = dataset['train']
      LOGGER.info(f'Original lm1b train set size: {len(full_train)} samples')
      
      # Use reproducible split with seed=42
      train_val_split = full_train.train_test_split(
        test_size=300000,  # 300K samples for validation
        seed=42,
        shuffle=True)
      
      # Save the splits to cache
      train_val_split['train'].save_to_disk(lm1b_train_cache)
      train_val_split['test'].save_to_disk(lm1b_val_cache)  # 'test' contains the validation split
      LOGGER.info(f'Split created - Train: {len(train_val_split["train"])} samples, Validation: {len(train_val_split["test"])} samples')
      
      if mode == 'train':
        data = train_val_split['train']
      else:  # mode == 'validation'
        data = train_val_split['test']
  elif dataset_name == 'qm9':
    data = dataset
  else:
    data = dataset[mode]

  if dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detoker):
    def detok(text):
      for j, t in enumerate(text, 0):
        text[j] = detoker(t)
      return text
    return detok

  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if 'amazon_polarity' in dataset_name:
      text = example['content']
    elif 'qm9' in dataset_name:
      text = example['canonical_smiles']
    elif dataset_name == 'ten_species':
      text = example['sequence']
    else:
      text = example['text']

    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      if add_special_tokens:
        tokens = {'input_ids':
                  [t + [EOS] for t in tokens['input_ids']]}
        # Still missing BOS; will be added in group_texts
      else:
        tokens = {'input_ids': tokens['input_ids']}
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=add_special_tokens,
                         return_attention_mask=True,
                         return_token_type_ids=add_special_tokens)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  keep_cols = ['input_ids', 'token_type_ids',
               'attention_mask']
  if label_col is not None:
    keep_cols.append(label_col)
  tokenized_dataset = tokenized_dataset.remove_columns(
    [col for col in tokenized_dataset.column_names
     if col not in keep_cols])

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS,
    add_special_tokens=add_special_tokens)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  elif config.data.tokenizer_name_or_path == 'raw_pixels':
    tokenizer = custom_datasets.discretized_cifar10.DummyVisionTokenizer(
      256, 32,
      add_mask_token=config.data.add_mask_token,
      add_special_tokens=config.data.add_special_tokens)
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path,
      trust_remote_code=True)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None and not config.is_vision:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None, force_val=False):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  label_col = getattr(config.data, 'label_col', None)
  if skip_train:
    train_set = None
  else:
    if 'cifar10' in config.data.train:
      train_set = custom_datasets.discretized_cifar10.DiscreteCIFAR10(
        config.data.train, train=True, download=True)
    else:
      train_set = get_dataset(
        config.data.train,
        tokenizer,
        mode='train',
        wrap=config.data.wrap,
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        override_cache=config.data.override_cache,
        add_special_tokens=config.data.add_special_tokens,
        label_col=label_col,
        label_threshold=getattr(config.data,
                                'label_col_pctile', None))
  if config.data.valid in [
    'text8', 'lm1b', 'amazon_polarity', 'qm9',
    'ten_species']:
    validation_split = 'test'
    if force_val and config.data.valid == 'lm1b':
      validation_split = 'validation'  # For lm1b, this uses the 300K samples split from train
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    if 'cifar10' in config.data.train:
      valid_set = custom_datasets.discretized_cifar10.DiscreteCIFAR10(
        config.data.valid, train=False, download=True)
    else:
      valid_set = get_dataset(
        config.data.valid,
        tokenizer,
        wrap=config.data.wrap,
        mode=validation_split,
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        streaming=False,
        override_cache=config.data.override_cache,
        add_special_tokens=config.data.add_special_tokens,
        label_col=label_col,
        label_threshold=getattr(config.data,
                                'label_col_pctile', None))

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=config.loader.persistent_workers
    )
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
