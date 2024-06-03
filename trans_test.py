import os
import re
import sys
import pandas as pd
from janome.tokenizer import Tokenizer
from collections import Counter
import PyPDF2
import requests

# ストップワードのリスト
STOP_WORDS = set([
    "する", "ある", "いる", "こと", "これ", "それ", "あれ", "ため", "もの", "ここ", "さん", "そう",
    "これら", "それら", "あれら", "及び", "および", "において", "さらに"
])

# 英単語を検出するための正規表現パターン
ENGLISH_WORD_PATTERN = re.compile(r'[a-zA-Z]+')


# PDFからテキストを抽出する関数
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages])


# 前処理を行う関数
def preprocess_text(text):
    text = re.split('底本：', text)[0]
    text = re.sub('一', '', text, 1)
    text = re.sub('_', '', text, 1)
    text = re.sub('《.+?》', '', text)
    text = re.sub('［＃.+?］', '', text)
    text = re.sub('\n\n', '\n', text)
    text = re.sub(r'\d+', '', text)  # 数字の削除
    text = re.sub(r'[^\w\s]', '', text)  # 記号の削除
    text = text.replace("\n", "").replace(" ", "")
    return text


# 形態素解析を行い、ストップワードおよび英単語を除いた単語を集計する関数
def analyze_text(text):
    tokenizer = Tokenizer()
    return [
        (token.base_form, token.part_of_speech.split(',')[0], token.reading)
        for token in tokenizer.tokenize(text)
        if token.base_form not in STOP_WORDS and not ENGLISH_WORD_PATTERN.
        match(token.base_form) and token.part_of_speech.split(',')[0] != "記号"
    ]


# DeepL APIを使用してテキストを翻訳する関数
def translate_text(text, target_lang, api_key):
    url = "https://api-free.deepl.com/v2/translate"
    params = {"auth_key": api_key, "text": text, "target_lang": target_lang}
    response = requests.post(url, data=params)
    response.raise_for_status()
    return response.json()["translations"][0]["text"]


# DataFrame内の単語を翻訳する関数
def translate_df(df, target_lang, api_key):
    if not isinstance(df, pd.DataFrame) or "単語" not in df.columns:
        raise ValueError(
            'Input must be a pandas DataFrame containing a "単語" column.')
    if "翻訳後" in df.columns:
        return df

    words = '\n'.join(df["単語"].tolist())
    translated_list = translate_text(words, target_lang,
                                     api_key).strip().split('\n')
    df["翻訳後"] = translated_list
    return df


# 主な処理
def main(pdf_path, api_key, target_lang):
    text = extract_text_from_pdf(pdf_path)
    text = preprocess_text(text)
    word_pos_list = analyze_text(text)
    word_freq = Counter([word for word, _, _ in word_pos_list])

    df = pd.DataFrame(word_pos_list, columns=["単語", "品詞", "ふりがな"])
    df["出現回数"] = df["単語"].map(word_freq)
    df = df.drop_duplicates(subset=["単語"]).sort_values(
        by="出現回数", ascending=False).reset_index(drop=True)

    translated_df = translate_df(df, target_lang, api_key)
    print(translated_df)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <pdf_file> <deepl_api_key> <target_lang>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    deepl_api_key = sys.argv[2]
    target_lang = sys.argv[3]

    main(pdf_file, deepl_api_key, target_lang)

