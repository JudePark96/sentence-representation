from konlpy.tag import Mecab
from tqdm import tqdm
import _pickle as p


def split_punctuation(path: str) -> None:

    file_name = 'wiki_preprocessed.txt'
    tokenizer = Mecab()
    result = []

    with open(f'{path}{file_name}', 'r') as f:
        for seqs in tqdm(f):
            seqs = seqs.split('.')

            for seq in seqs:
                result.append(tokenizer.morphs(seq.strip()))
        f.close()

    with open(f'{path}split_punc.pkl', 'wb') as f:
        print('start dumping pickle')
        p.dump(result, f)


if __name__ == '__main__':
    split_punctuation('./rsc/data/')
    pass