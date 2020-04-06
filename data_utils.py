__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from transformers import BertTokenizer


def tokenize(tokenizer: BertTokenizer, sentence: str, max_len: int = 50):
    # For single sequences:
    #  tokens:   [CLS] I am a boy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0

    tokens = tokenizer.encode_plus(sentence, max_length=max_len, pad_to_max_length=True)
    assert len(tokens['input_ids']) == max_len
    assert len(tokens['token_type_ids']) == max_len
    assert len(tokens['attention_mask']) == max_len

    return tokens


if __name__ == '__main__':
    tokenize(BertTokenizer.from_pretrained('bert-base-uncased'), 'the dog is hairy .')
    pass