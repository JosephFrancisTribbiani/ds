import json
import spacy
from bs4 import BeautifulSoup


def parse_text(text: str, tag: str = "person") -> str:
    soup = BeautifulSoup(text, "lxml-xml")
    return soup.find(tag).text


def _check_token(token: spacy.tokens.token.Token, nlp: spacy.lang, pos: list = None) -> bool:
    return ((token.pos_ in pos) if pos is not None else True) & \
           (token.lemma_ not in nlp.Defaults.stop_words) & \
           (len(token.lemma_) > 1) & \
           ~(token.lemma_.isdigit())


def data_preprocessing(*corp, nlp: spacy.lang, pos: list = None, n_process: int = 1, filename: str = None):

    # iterate over texts and preprocess them
    for texts in corp:
        texts_preprocessed = [[token.lemma_ for token in doc if _check_token(
            token=token, nlp=nlp, pos=pos)] for doc in nlp.pipe(texts, n_process=n_process)]
        
        if filename:
            with open(filename, "a", encoding="utf-8") as file:
                
                # dump and save to file
                json_data = json.dumps(texts_preprocessed, ensure_ascii=False)
                file.write(json_data + "\n")
