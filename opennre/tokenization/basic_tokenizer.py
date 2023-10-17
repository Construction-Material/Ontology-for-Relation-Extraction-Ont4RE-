# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BasicTokenizer classes.一个基本的分词器，将文本分词为单词和标点符号"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .utils import (convert_to_unicode,
                   clean_text,
                   split_on_whitespace,
                   split_on_punctuation,
                   tokenize_chinese_chars,
                   strip_accents,
                    jieba_tokenization)

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.).运行基本的标记化（标点符号拆分、小写字母等）。"""

    def __init__(self, 
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        # text = jieba_tokenization(text)
        text = convert_to_unicode(text)
        text = clean_text(text)
        text = tokenize_chinese_chars(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.)这是在2018年11月1日为多语言和中文模型添加的。
        # 现在这也适用于英文模型，但这并不重要，因为英文模型没有在任何中文数据上进行训练，
        # 而且一般来说，其中没有任何中文数据（词汇中存在汉字，因为维基百科中确实有一些中文词汇）.
        orig_tokens = split_on_whitespace(text)
        split_tokens = []
        current_positions = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = strip_accents(token)
            current_positions.append([])
            current_positions[-1].append(len(split_tokens))
            split_tokens.extend(split_on_punctuation(token))
            current_positions[-1].append(len(split_tokens))
        return split_tokens, current_positions
