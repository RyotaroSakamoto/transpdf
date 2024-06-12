import streamlit as st
import re
import json
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

#翻訳先の言語辞書を読み込み
with open("languages.json") as f:
    languages = json.load(f)

lang_names = [language["name"] for language in languages]
lang_dict = {language["name"]: language["language"] for language in languages}


# PDFからテキストを抽出する関数
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
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


# Streamlitアプリケーション
st.set_page_config(layout="wide")  # ページレイアウトをワイドに設定

st.title("PDF Word Translation/PDF単語翻訳")
st.write("Aggregate and translate PDF files written in Japanese word by word")
st.write("日本語で書かれたPDFファイルを単語ごとに集計して翻訳します")
st.markdown(
    "[How to get Deepl's API key/Deepl APIkeyの取得方法](https://zenn.dev/eito_blog/articles/2e353b96a42494)"
)

pdf_file = st.file_uploader("Upload PDF file/PDFファイルをアップロード", type="pdf")
deepl_api_key = st.text_input(
    "Please enter your API key for DeepL/DeepL APIキーを入力してください")
option = st.selectbox(
    "Please select the language you wish to translate into/翻訳先の言語を選択してください",
    lang_names,
    index=6)
target_lang = "en-us"  #default language
target_lang = lang_dict[option].lower()

if st.button("実行"):
    if pdf_file is not None and deepl_api_key and target_lang:
        text = extract_text_from_pdf(pdf_file)
        text = preprocess_text(text)
        word_pos_list = analyze_text(text)
        word_freq = Counter([word for word, _, _ in word_pos_list])

        df = pd.DataFrame(word_pos_list, columns=["単語", "品詞", "ふりがな"])
        df["出現回数"] = df["単語"].map(word_freq)
        df = df.drop_duplicates(subset=["単語"]).sort_values(
            by="出現回数", ascending=False).reset_index(drop=True)
        translated_df = translate_df(df, target_lang, deepl_api_key)
        translated_df = translated_df.reindex(
            columns=["出現回数", "単語", "ふりがな", "翻訳後", "品詞"])

        # テーブル表示を最大化
        st.dataframe(translated_df, hide_index=True, height=1000, width=500)
    else:
        st.error(
            "Please make sure all fields are filled out correctly./すべての入力項目を正しく入力してください。"
        )
