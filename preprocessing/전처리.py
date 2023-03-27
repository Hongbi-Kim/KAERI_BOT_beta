from io import BytesIO
from PyPDF2 import PdfReader # pdf 내용 추출
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import openai
import os, re, tenacity, pickle, unicodedata
from pykospacing import Spacing

# 원숫자 ① -> 1.
def circled_to_normal(char):
    circled_numbers = u"\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469"
    normal_numbers = "1234567890"
    
    if char in circled_numbers:
        return normal_numbers[circled_numbers.index(char)] + "."
    return char

class Chatbot():
    
    # PDF 텍스트 분석 및 처리
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        blob_text = ''
        processed_text = []
        
        for i in range(number_of_pages):
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                text = "".join([circled_to_normal(c) for c in text]) # 원문자
                text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9!@#$%^&*(),.?~"-:{}<>%/\\]', '', text.strip())
                x = tm[4]
                y = tm[5]
                page_text.append({
                'fontsize': fontSize,
                'text': text, #.strip().replace('\x03', ''),
                'x': x,
                'y': y
                })

            _ = page.extract_text(visitor_text=visitor_body) # 페이지 텍스트 추출


            for t in enumerate(page_text):
                t = t[1]
                if t['text'] != '':
                    blob_text += f" {t['text']}"
                    processed_text.append({
                        'fontsize': t['fontsize'],
                        'text': t['text'],
                        'page': i
                    })
        # 정규 표현식 패턴 정의
        pattern = r"\d{4}/\d{2}/\d{2}\s+(.+?)\s+\("

        # 패턴에 맞는 문자열 찾기
        match = re.search(pattern, blob_text)

        title = match.group(1)
            
        blob_text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", blob_text)
        blob_text = re.sub(r"제\s*(\d+)\s*장", r"제\1장", blob_text)
        blob_text = re.sub(r"부 칙", r"부칙", blob_text)
        
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
        
        # 문서 제목 앞 문서 번호? 삭제
        pattern2 = r'^.*?'+title
        blob_text = re.sub(pattern2, title, blob_text)
        
        blob_text = blob_text.replace(' ','') # 띄어쓰기 없애기
        
        # 형태소 분석 결과를 이용해 띄어쓰기를 적용한 문자열 생성
        spacing = Spacing(rules=[title])
        blob_text = spacing(blob_text)
        
        # 패턴 적용해서 결과 출력
        blob_text = re.sub(r'\([^)]*\)', ' ', blob_text) # (괄호 내용) 삭제
        blob_text = re.sub(r'\<[^)]*\>', ' ', blob_text)
        blob_text = re.sub(r'부칙.*', '', blob_text)
        blob_text = re.sub(r"(\S) 라(\S)한다", r"\1라 \2한다", blob_text)
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) +\. ?', r'\1. ', blob_text)
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) +\, ?', r'\1, ', blob_text)
        blob_text = re.sub(r'(\d)([ㄱ-ㅎㅏ-ㅣ가-힣])', r'\1\2', blob_text) # 숫자+한글 붙이기
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣])(\d)', r'\1\2', blob_text)
        pattern3 = r'\.(\d+)' # .+숫자 띄기
        blob_text = re.sub(pattern3, r'. \1', blob_text)
        pattern4 = r'(\d+)\. (\d+)' # 숫자+.+숫자 붙이기
        blob_text = re.sub(pattern4, r'\1.\2', blob_text)
        pattern5 = r"(제\d+조)(\d+) *(\.)" # 조 1
        blob_text = re.sub(pattern5, r"\1 \2 ", blob_text)
        pattern6 = r'\.([ㄱ-ㅎㅏ-ㅣ가-힣])' # .+한글 띄기
        blob_text = re.sub(pattern6, r'. \1', blob_text)
        pattern7 = r'\.(\d+)\.' # .1. -> . 1. 붙이기
        blob_text = re.sub(pattern7, r'. \1.', blob_text)
        pattern8 = r'(\d+)\, (\d+)' # 숫자+,+숫자 붙이기
        blob_text = re.sub(pattern8, r'\1,\2', blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text)
        blob_text = re.sub(r"목차", r"", blob_text)
        paper_text = blob_text
        print("Done parsing paper")
        # print(paper_text)
        return paper_text

    def paper_df(self, pdf): # pdf를 분석하여 df로 변환
        print('Creating dataframe')
        
        p1 = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!) +'
        sentences = re.split(p1, pdf)
        
        ss = ''
        ss2 = []
        for s in sentences:
            if len(ss)<500: # ss이 500글자가 안넘으면 
                ss = (ss + ' ' +s) # s 문장 포함
                if len(ss)>500: # 포함했는데 ss이 500글자가 넘으면
                    ss2.append(ss) # ss2로 데이터프레임 열에 추가
            else: #ss가 500글자가 넘으면
                ss = '' # ss 초기화
                ss = (ss + ' ' +s) # s 문장 포함
        if len(ss)>300: # 마지막 남은 ss가 300글자 이상이라면 ss2에 포함 
            ss2.append(ss)
        else:
            if ss2 != []:
                ss2[-1] = ss2[-1] + ss
            else:
                ss2.append(ss)
        df = pd.DataFrame(ss2)
        return df

print("Processing pdf")
chatbot = Chatbot()
openai.api_key = "your openai key" 
pdf_folder = '규정' # PDF 파일이 있는 폴더 경로
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf') and f.startswith('원규')]
df_all = pd.DataFrame(columns=['text', 'CLS1', 'paper_title', 'embeddings'])
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

    openai.api_key = "your openai key" 
    embedding_model = "text-embedding-ada-002"
    embeddings = df.iloc[:,0].apply([lambda x: get_embedding(x, engine=embedding_model)])
    df["embeddings"] = embeddings

    df_all = pd.concat([df_all, df])

df = df_all.copy()
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)
print("Done processing pdf")