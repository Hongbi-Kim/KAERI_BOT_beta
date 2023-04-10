from io import BytesIO
from PyPDF2 import PdfReader # pdf 내용 추출
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import openai
import os, re, tenacity, pickle, unicodedata
from pykospacing import Spacing
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
okt = Okt()

# 원숫자 ① -> 1., o->ㅇ, △-> ㅁ
def circled_to_normal(char):
    circled_numbers = u"\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469\u25CB\u25B3"
    normal_numbers = "1234567890ㅇㅁ"
    
    if char in circled_numbers[:-2]:
        return normal_numbers[circled_numbers.index(char)] + "."
    if char in circled_numbers[-2:]:
        return normal_numbers[circled_numbers.index(char)]
    return char

class Chatbot():
    
    # PDF 텍스트 분석 및 처리
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        blob_text = ''
        
        for i in range(number_of_pages):
            page_lines = ''
            page = pdf.pages[i]

            def visitor_body(text, cm, tm, fontDict, fontSize):
                global page_lines
            page_lines = page.extract_text(visitor_text=visitor_body) # 페이지 텍스트 추출
            blob_text += page_lines
            
        blob_text = re.sub(r'\n', '', blob_text)
        blob_text = "".join([circled_to_normal(c) for c in blob_text]) # 원문자, ○△
        blob_text = blob_text.replace('\u0027', '') # '
        blob_text = blob_text.replace('\u201C', '') # “
        blob_text = blob_text.replace('\u201D', '') # ”
        blob_text = blob_text.replace('\uFF05', '%') # ％ -> %
        blob_text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9!@#$%￦^&*(),.-~:{}<>/\\]', ' ', blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로

        blob_text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", blob_text)
        blob_text = re.sub(r"제\s*(\d+)\s*장", r"제\1장", blob_text)
        
        pattern1 = re.compile(r'목차[\s\S]*제1장') # '목차'로 시작하고 '제1장'이 나오는 패턴
        blob_text = pattern1.sub('제1장', blob_text)

        start = blob_text.find('목차') # '목차'로 시작하고, '제1조'이 나오는 패턴
        # match = re.search(r'(제1조\([^)]+\)).*?(제1조\([^)]+\))', blob_text)
        match = re.search(r'(제1조).*?(제1조)', blob_text)
        
        if start != -1 and match:
            end = match.start(2)
            blob_text = blob_text[:start] + blob_text[end:]
        else:
            pass

        blob_text = re.sub(r'\([^)]*\)', ' ', blob_text) # (괄호 내용) 삭제
        blob_text = re.sub(r'\<[^)]*\>', ' ', blob_text) # <내용> 삭제
        
        # blob_text = blob_text.replace(' ','') # 띄어쓰기 없애기
        
        # # 형태소 분석 결과를 이용해 띄어쓰기를 적용한 문자열 생성
        # spacing = Spacing(rules=[title])
        # blob_text = spacing(blob_text)

        
        # 패턴 적용해서 결과 출력
        if blob_text[18:24] == '원규관리규정':
            blob_text = re.sub(r'부칙부 칙.*', '', blob_text)
            blob_text = re.sub(r"부 칙", r"부칙", blob_text)
        else:
            blob_text = re.sub(r"부 칙", r"부칙 ", blob_text)
            blob_text = re.sub(r'부칙 .*', '', blob_text) # 부칙 ~ 별칙 삭제 -> 원규관리규정 문서에서 적용xx.....
        # blob_text = re.sub(r"부 칙", r"부칙", blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) +\. ?', r'\1. ', blob_text) # 마침표 붙이기
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) +\, ?', r'\1, ', blob_text) # , 붙이기 
        # 문서 번호 및 날짜 
        pattern3 = r"\d{4}/\d{2}/\d{2}"
        blob_text = re.sub(f".*{pattern3}", "", blob_text)
        blob_text = re.sub(r"목차", r"", blob_text)
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) (\의 |\로써 |\한다 |\은 |\을 |\를 |\이란 |\란 |\에 |\는 |\가 |\이라 |\라 |\이고 |\로 |\에서 |\에도 |\에게 |\야 |\과 )', r'\1\2', blob_text)
        blob_text = re.sub(r"결 산", r'결산', blob_text)
        blob_text = re.sub(r"기록 부", r'기록부', blob_text)
        blob_text = re.sub(r"재물조사 서", r'재물조사서', blob_text)
        blob_text = re.sub(r"절 차", r'절차', blob_text)
        blob_text = re.sub(r"소 정", r'소정', blob_text)
        blob_text = re.sub(r"연 구원", r'연구원', blob_text)
        blob_text = re.sub(r"석 사", r'석사', blob_text)
        blob_text = re.sub(r"박 사", r'박사', blob_text)
        blob_text = re.sub(r"정 함", r'정함', blob_text)
        blob_text = re.sub(r"체 계적인", r'체계적인', blob_text)
        blob_text = re.sub(r"계 획", r'계획', blob_text)
        blob_text = re.sub(r"보 증", r'보증', blob_text)
        blob_text = re.sub(r" 조 직 ", r' 조직 ', blob_text)
        blob_text = re.sub(r"여 부", r'여부', blob_text)
        blob_text = re.sub(r"채 용", r'채용', blob_text)
        blob_text = re.sub(r"승 진", r'승진', blob_text)
        blob_text = re.sub(r"급 여", r'급여', blob_text)
        blob_text = re.sub(r"겸 직", r'겸직', blob_text)
        blob_text = re.sub(r"복 무", r'복무', blob_text)
        blob_text = re.sub(r"삭 제", r'삭제', blob_text)
        blob_text = re.sub(r"보 칙", r'보칙', blob_text)
        blob_text = re.sub(r"교 육", r'교육', blob_text)
        blob_text = re.sub(r"승 격", r'승격', blob_text)
        blob_text = re.sub(r"구 성 원", r'구성원', blob_text)
        blob_text = re.sub(r"삭제", r'', blob_text)
        pattern4 = r'\.(\d+)\.' # .1. -> . 1. 띄우기
        blob_text = re.sub(pattern4, r'. \1.', blob_text)
        pattern5 = r'\.([ㄱ-ㅎㅏ-ㅣ가-힣])' # .+한글 띄기
        blob_text = re.sub(pattern5, r'. \1', blob_text)
        pattern6 = r'([ㄱ-ㅎㅏ-ㅣ가-힣])(\d+)\.' # 한글+숫자. 띄기
        blob_text = re.sub(pattern6, r'\1 \2.', blob_text)
        pattern7 = r'\.(제)' # 다.제1조 -> 다. 제1조
        blob_text = re.sub(pattern7, r'. \1', blob_text)
        pattern8 = r'([ㄱ-ㅎㅏ-ㅣ가-힣])(제\d+)' # 경우제1 -> 경우 제1
        blob_text = re.sub(pattern8, r'\1 \2', blob_text)
        blob_text = re.sub('-', ' - ', blob_text)
        blob_text = re.sub(r'\)', '', blob_text)
        blob_text = re.sub(r'\(', '', blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        pattern9 = r'(\d+)\. (\d+)' # 숫자+.+숫자 붙이기
        blob_text = re.sub(pattern9, r'\1.\2', blob_text)
        pattern10 = r"(제\d+조)(\d+)" # 제(숫자)조1 -> 제(숫자)조 1
        blob_text = re.sub(pattern10, r"\1 \2 ", blob_text)
        pattern11 = r"(제\d+호의)(\d+)" # 제(숫자)호의1 -> 제(숫자)호의 1
        blob_text = re.sub(pattern11, r"\1 \2 ", blob_text)
        blob_text = re.sub(r'\.{2,}', '. ', blob_text) ## .. -> .
        blob_text = re.sub(r'\. +\.', '. ', blob_text) ## . . -> .
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        
        if blob_text[-1] == ' ': 
            blob_text = blob_text[:-1]
        if blob_text[-2:] == '부칙': 
            blob_text = blob_text[:-2]
            
        paper_text = blob_text
        print("Done parsing paper")
        # print(paper_text)
        return paper_text

    def paper_df (self, pdf): # pdf를 분석하여 df로 변환
        print('Creating dataframe')
        
        # 문장들을 잘라서 리스트에 담기
        sentences = re.split(r'(?<=\s)(?=제\d+조\s|제\d+장\s)', pdf)
        # 각 문장의 앞뒤 공백 제거
        sentences = [s.strip() for s in sentences if not re.match(r'^제\d+조\s*$', s) and not re.match(r'^제\d+장\s*$', s.strip())]

        # 잘린 문장들을 하나씩 가져와서 처리하기
        ss = ''  # ss: 앞서 가져온 문장들을 합쳐서 담을 변수
        ss2 = []  # ss2: 최종적으로 만들어진 문장들을 담을 리스트
        
        for s in sentences:
            s = s.strip()  # 각 문장의 앞뒤 공백 제거
            if len(s) >= 400:  # 문장이 400글자 이상이면
                if len(ss) > 250:  # 이전까지 모아진 ss가 250글자 이상이면
                    ss2.append(ss.strip())  # ss를 ss2에 추가
                    ss = ''
                    ss2.append(s)  # 현재 문장도 ss2에 추가
                else:  # 이전까지 모아진 ss가 250글자 미만이면
                    if ss2 != []: 
                        if len(ss2[-1]) < len(s):  # 마지막 문장이 현재 문장보다 길이가 작으면
                            ss2[-1] += ' ' + ss.strip()  # 마지막 문장에 ss를 추가
                            ss = ''
                            ss2.append(s)  # 현재 문장도 ss2에 추가
                    else:
                        ss2.append(ss.strip() + ' ' + s)  # ss와 현재 문장을 합쳐서 ss2에 추가
                        ss = ''
            else:  # 문장이 400글자 미만이면
                if len(ss) > 250:  # 이전까지 모아진 ss가 250글자 이상이면
                    ss2.append(ss.strip())  # ss를 ss2에 추가
                    ss = s
                else:
                    ss += (' ' + s)  # 현재 문장을 ss에 추가하기

        if len(ss) > 0:  # 마지막으로 모아진 ss가 있다면
            if ss2 != []:
                if len(ss) < 250:  # 마지막으로 모아진 ss와 현재 문장의 길이가 250글자 미만이면
                    ss2[-1] += ' ' + ss.strip()  # 마지막 문장에 ss를 추가
                else:
                    ss2.append(ss.strip())  # ss를 ss2에 추가
            else:
                ss2.append(ss.strip())  # ss를 ss2에 추가

        df = pd.DataFrame(ss2)  # 최종적으로 만들어진 문장들을 데이터프레임으로 변환

        return df

    def paper_df_num(self, pdf): # 추가 처리 -> 약 300글자 넘지 않게
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


print("Processing pdf")
chatbot = Chatbot()
openai.api_key = "your openai key" 
pdf_folder = '규정' # PDF 파일이 있는 폴더 경로
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf') and not f.startswith('X')]
df_all = pd.DataFrame(columns=['text', 'CLS1', 'paper_title'])

######################################################################################################
#### case 1 : cosine similary
######################################################################################################
for p_f in pdf_files:
    with open('규정/'+p_f, 'rb') as f:
        file = f.read()
    pdf = PdfReader(BytesIO(file))
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df.columns = ['text']
    df['CLS1'] = [p_f.split('_')[1]]*len(df)
    df['paper_title'] = [p_f.split('_')[2].split('[')[0]]*len(df)
    df_all = pd.concat([df_all, df])
df = df_all.copy()
df.reset_index(drop=True, inplace=True)

df = chatbot.paper_df_num(df)
df['text'] = df['text'].map(lambda x: '['+ p_f.split('_')[2].split('[')[0]+ '] ' + x.strip())
embedding_model = "text-embedding-ada-002"
embeddings = df.iloc[:,0].apply([lambda x: get_embedding(x, engine=embedding_model)])
df["embeddings"] = embeddings
print("Done processing pdf")

######################################################################################################
#### case 1 : bm25
######################################################################################################
def tokenizer(sent):
  sent = okt.morphs(sent, norm=False, stem=True)
  return sent

pdf_folder = '규정' # PDF 파일이 있는 폴더 경로
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf') and not f.startswith('X')]
df_all = pd.DataFrame(columns=['text', 'CLS1', 'paper_title'])
for p_f in pdf_files:
    with open('규정/'+p_f, 'rb') as f:
        file = f.read()
    pdf = PdfReader(BytesIO(file))
    paper_text = chatbot.parse_paper(pdf)
    # global df
    df = chatbot.paper_df(paper_text)
    df.columns = ['text']
    df['CLS1'] = [p_f.split('_')[1]]*len(df)
    df['paper_title'] = [p_f.split('_')[2].split('[')[0]]*len(df)
    df['text'] = df['text'].map(lambda x: '['+ p_f.split('_')[2].split('[')[0]+ '] ' + x.strip())
    df_all = pd.concat([df_all, df])

df = df_all.copy()
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)

embeddings = [tokenizer(doc) for doc in df.iloc[:,0]]
bm25 = BM25Okapi(embeddings)

# 저장
# with open('정제파일/원예규A_bm25_model.pkl', 'wb') as f:
#     pickle.dump(bm25, f)