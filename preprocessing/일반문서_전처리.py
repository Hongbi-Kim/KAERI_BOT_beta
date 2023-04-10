from io import BytesIO
from PyPDF2 import PdfReader # pdf 내용 추출
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import openai
import os, re, tenacity, pickle, unicodedata
import pdfplumber
from konlpy.tag import Okt


# 원숫자 ① -> 1., o->ㅇ, △-> ㅁ
def circled_to_normal(char):
    circled_numbers = u"\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469"
    normal_numbers = "1234567890"
    
    if char in circled_numbers[:-2]:
        return normal_numbers[circled_numbers.index(char)] + "."
    if char in circled_numbers[-2:]:
        return normal_numbers[circled_numbers.index(char)]
    return char
     
def paper_df_조(pdf): # pdf를 분석하여 df로 변환
    print('Creating dataframe')

    sentences = re.split(r'(?<=\s)(?=제\d+조\*|제\d+장\s)', pdf)
    sentences = [s.strip() for s in sentences if not re.match(r'^제\d+조\**$', s) and not re.match(r'^제\d+장\s*$', s.strip())]

    ss = ''
    ss2 = []

    for s in sentences:
        s = s.strip()
        if len(s) >= 600:  # s가 400글자 이상일 때
            if len(ss) > 400:  # 이전까지 모아진 ss가 있다면 ss2에 추가
                ss2.append(ss.strip())
                ss = ''
                ss2.append(s) # s도 추가
            else: # ss가 250이하라면
                if ss2 != []: 
                    if len(ss2[-1]) < len(s): # 작은 곳에 ss추가
                        ss2[-1] += ' ' + ss.strip()
                        ss = ''
                        ss2.append(s)
                else:
                    ss2.append(ss.strip() + ' ' + s) # s를 ss2에 추가
                    ss = ''
        else: # s가 400글자 미만일 때
            if len(ss)>400:
                ss2.append(ss.strip())
                ss = s
            else:
                ss += (' ' + s)

    if len(ss) > 0:
        if ss2 != []:
            if len(ss) < 400:
                ss2[-1] += ' ' + ss.strip()
            else:
                ss2.append(ss.strip())
        else:
            ss2.append(ss.strip())
    df = pd.DataFrame(ss2)

    return df

def paper_df_num(pdf):
    pdf = sum_df
    pdf = list(filter(lambda x: x != "", pdf))
    for n, p in enumerate(pdf):
        if n == len(pdf)-1 and len(p)<250: # 마지막 페이지가 너무 적을 때
            pdf[n-1] = pdf[n-1] + p
            pdf[n] = ''
        if n != len(pdf)-1 and len(p)>560:
            pdf[n+2:] = pdf[n+1:-1]
            pdf.append(pdf[-1])
            pdf[n] = p[:int(len(p)/2)]
            pdf[n+1] = p[int(len(p)/2):]
    df = pd.DataFrame(pdf)
    df = df.drop(df[df.apply(lambda row: row.str.strip().eq('')).any(axis=1)].index).reset_index(drop=True)
    return df



def process_text(text):
    text = text.replace('목 차', '목차')
    text = text.replace('\u00B7', ' ') # -> ·
    text = text.replace('\u25CB', ' ') # -> ○
    text = "".join([circled_to_normal(c) for c in text])
    text = text.replace('\u25C8', '^') # -> ◈
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9!@#$%￦^&*(),.-~:{}<>/\\]', ' ', text)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) (\의 |\로써 |\한다 |\은 |\을 |\를 |\이란 |\란 |\에 |\는 |\가 |\이라 |\라 |\이고 |\로 |\에서 |\에도 |\에게 |\야 |\과 )', r'\1\2', text)
    text = re.sub(r"조제", r'조 제', text)
    text = re.sub(r"불 구", r'불구', text)
    text = re.sub(r"교 류", r'교류', text)
    text = re.sub(r"관 한", r'관한', text)
    text = re.sub(r"수 정 공 시", r'수정공시', text)
    text = re.sub(r'\[[^)]*\]', ' ', text)
    text = re.sub(r'\<[^)]*\>', ' ', text)
    text = re.sub(r'\([^)]*\)', '*', text)
    text = text.replace('^', '\u25C8')
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def paper_df_num2(pdf):
    for n, p in enumerate(pdf.iloc[:, 0]):
        if n != len(pdf.iloc[:, 0]) - 1 and len(p) > 500:
            tt_l = ''
            p_split = re.split(r'(?<=다.)\s', p)
            num_splits = 0

            if len(p) < 600:
                num_splits = 2
            elif len(p) < 900:
                num_splits = 3
            else:
                num_splits = 4

            pdf_splits = pd.DataFrame(columns=pdf.columns)
            split_lengths = [int(len(p) / num_splits) for _ in range(num_splits - 1)] + [len(p) - sum(int(len(p) / num_splits) for _ in range(num_splits - 1))]

            for i, split_length in enumerate(split_lengths):
                split_text = ' '.join(p_split[:split_length])
                p_split = p_split[split_length:]
                pdf_splits.loc[i] = pdf.loc[n]
                pdf_splits.iloc[i, 0] = split_text

            pdf = pd.concat([pdf.iloc[:n], pdf_splits, pdf.iloc[n + 1:]]).reset_index(drop=True)
    pdf = pdf[pdf.iloc[:, 0] != ''].reset_index(drop=True)

    df = pdf
    return df

#### 전처리 시작 ####

print("Processing pdf")
openai.api_key = "your openai key" 
pdf_folder = 'Alio/중요' # PDF 파일이 있는 폴더 경로
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
df_all = pd.DataFrame(columns=['text', 'CLS1', 'paper_title'])

for pdf_file in pdf_files:
    sum_df = []
    blob_text = ''
    with pdfplumber.open('Alio/중요/'+pdf_file) as pdf:
        for page_number, page in enumerate(pdf.pages):
            print("------------------------------")
            print(f"Page {page_number + 1}:")
            sum_text = ''
            
            # 페이지의 텍스트 블록을 순회합니다.
            for item in page.extract_words():
                text = list(item.values())[0]
                sum_text += (text + ' ')
                if sum_text[:2] == '목차': # 목차 페이지 삭제
                    sum_text = ''
            blob_text += (sum_text + ' ')
            sum_df.append(process_text(sum_text))
            blob_text = process_text(blob_text)

    if len(paper_df_조(blob_text)) == 1:
        df = paper_df_num(sum_df)
    else:
        df = paper_df_조(blob_text)

    df[0] = df[0].apply(lambda x: x.replace("*", ""))
    df.columns = ['text']
    df['CLS1'] = [""]*len(df)
    df['paper_title'] = [pdf_file[:-4]]*len(df)
    df_all = pd.concat([df_all, df])
df = df_all.copy()
df.reset_index(drop=True, inplace=True)

df = paper_df_num2(df)
df['text'] = df.apply(lambda x: '[' + x['paper_title'] + '] ' + x['text'].strip(), axis=1)

openai.api_key = "your openai key" 
embedding_model = "text-embedding-ada-002"
embeddings = df.iloc[:,0].apply([lambda x: get_embedding(x, engine=embedding_model)])
df["embeddings"] = embeddings

#### 유사도 사전 계산 ####
## 규정문서_전처리 참고 ##
