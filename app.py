## 본 코드는 현재 파동이봇 코드와는 약간 다릅니다.
## 전체적인 처리 과정은 비슷하니 참고용으로 봐주세요.

import streamlit as st
import json
import pandas as pd
import numpy as np
import openai, uuid, time
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import os, re, tenacity, pickle

# DB 저장 라이브러리
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

# bm25 관련 라이브러리
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
okt = Okt()

openai.api_key = "your openai key"

# 환경 변수에서 Google Cloud 서비스 계정 키를 가져옵니다.
service_account_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# 인증서 객체를 생성합니다.
with open(service_account_key) as json_file:
    cred = credentials.Certificate(json.load(json_file))

# Google Cloud Firestore 초기화
try:
    app = firebase_admin.initialize_app(cred)
except:
    pass
db = firestore.client()

# 평가 및 Firestore에 저장하는 함수
def save_to_firestore(chat):
    doc_ref = db.collection("conversations").document()
    data = {
        "user_question": chat["user"],
        "bot_answer": chat["bot"],
        "documents": chat["doc"].to_dict(orient='records'),
        "rating": chat['rating'],
    }
    doc_ref.set(data)


col1, col2 = st.columns([1,8])
with col1:
    st.image('images/jpg/mc파동이-100.png', width = 100)
with col2:
    st.title('파동이봇')
st.markdown('')

# 관련 데이터 불러오기
with open('정제파일/원예규_기본_Alio_최소.pkl', 'rb') as f:
    df_500 = pickle.load(f)
with open('원규_note_up150_300.pkl', 'rb') as f:
    df_300 = pickle.load(f)
with open('정제파일/원예규A_bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
def search_embeddings(df, query, n=4, pprint=True):
    ## cosine
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    results_co = df.sort_values("similarity", ascending=False, ignore_index=True).iloc[:2,:]

    ## bm25 ##
    def tokenizer(sent):
        sent = okt.morphs(sent, norm=False, stem=True)
        return sent
    tokenized_query = tokenizer(query)
    scores = bm25.get_scores(tokenized_query)

    top_n = np.argsort(scores)[::-1][:2]
    results_bm = df.loc[top_n]

    df_all = pd.concat([results_co, results_bm])
    df_all = df_all.drop_duplicates(subset=['text'])
    df_all.reset_index(drop=True, inplace=True)

    results = df_all
    # results = results_co
    return results.head(3)

def create_prompt(df, query, conversation):
    result = search_embeddings(df, query)
    
    ## 초기 프롬프트입니다.
    ## 현재 파동이의 프롬프트와는 다름.
    system_role = f"""You are an AI language model named "파동이" whose expertise is reading and summarizing Korea Atomic Energy Research Institute regulatory documents. 
    You must take the given embeddings and return a very detailed summary of the document in the language of the query. 
    Your conversation history is as follows:

        {conversation}
                
        Here are the embeddings: 
                
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """
        
        You must return in Korean. Please answer using polite and formal.
        Return a accurate answer based on the document and conversation history.
        If the question is ambiguous, refer to the previous questions and return.
        """

    user_content = f"""Given the question: "{str(query)}". """
    
    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]

    return messages

def gpt(messages):
    print('Sending request to GPT-3')
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0.4, 
        max_tokens=500)
    answer=(
            r["choices"][0]
            .get("message")
            .get("content")
            .encode("utf8")
            .decode()
        )
    print('Done sending request to GPT-3')
    response = {'answer': answer}
    print(response['answer'])
    return response

def checkboxes_changed(): # checkbox를 선택해도, 답변이 재생성 되지 않게
    return any(st.session_state.get(f"rat{i}_{len(st.session_state.conversation_history)}", False) for i in range(1, 4))

if 'chat_history' not in st.session_state:
    st.session_state.conversation_history = []
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'ratings' not in st.session_state:
    st.session_state.ratings = []
    
# 체크박스 초기화
if "checkboxes" not in st.session_state:
    st.session_state.checkboxes = {"rat1": False, "rat2": False, "rat3": False}
    
def update_ratings(rating):
    st.session_state.chat_history[-1]['rating'] = rating
    if "rating_done" not in st.session_state:
        st.session_state.rating_done = []
    st.session_state.rating_done.append(len(st.session_state.conversation_history) - 1)

# 체크박스를 클릭하면 평가 정보를 업데이트하는 함수를 추가합니다.
def check_if_checkboxes_changed():
    changed = False
    if st.session_state.get(f"rat1_{len(st.session_state.conversation_history)}"):
        changed = True
        update_ratings("불만족")
    elif st.session_state.get(f"rat2_{len(st.session_state.conversation_history)}"):
        changed = True
        update_ratings("보통")
    elif st.session_state.get(f"rat3_{len(st.session_state.conversation_history)}"):
        changed = True
        update_ratings("만족")
        
    return changed

conversation = ""

def handle_partial_response(choice):
    partial_text = choice["message"].get("content").encode("utf8").decode()
    st.write(partial_text)

# 이미지와 text_input을 함께 출력합니다.
col1, col2 = st.columns([1,8])
with col1:
    st.image('images/jpg/궁금-100.png', width = 90)
with col2:
    placeholder_text = '예시1: 휴가 결재는 누구한테 받아야 돼?\n예시2: 연구원 급여지급일이 언제야?\n예시3: 연구원 원장님 성함 알려줘'
    query = st.text_area("규정에 관련된 질문을 해주세요.:sunglasses: SEND 버튼 또는 Ctrl+Enter를 누르면 답변이 생성됩니다.", "", placeholder=placeholder_text)


# 체크박스 변경 여부를 확인하고 평가를 업데이트합니다.
check_if_checkboxes_changed()
checkboxes_changed = check_if_checkboxes_changed()

if (
    (st.button("SEND", key='message3') or (len(query) > 1)) and
    not checkboxes_changed
):

    # 이전 채팅이 있다면 저장합니다.
    try:
        save_to_firestore(st.session_state.chat_history[-1])
    except:
        pass
    st.session_state.conversation_history.append(query)
        
    # 대화 업데이트 및 GPT-3 메시지 전송
    if st.session_state.conversation_history:
        for i, chat in enumerate(st.session_state.conversation_history[:-1]):
            conversation += f"User: {chat}\n"
            if i < len(st.session_state.conversation_history) - 2:
                conversation += f"Assistant: {st.session_state.chat_history[i]['bot']}\n"
        conversation += f"User: {st.session_state.conversation_history[-1]}\n"
    prompt = create_prompt(df_500, query, st.session_state.conversation_history)
    
    # GPT-3 응답 처리
    response = gpt(prompt)
    answer = response['answer']
    
    # await st.experimental_run_async(gpt, prompt, handle_partial_response)
        
    # print(answer)
    paper = search_embeddings(df_500, query)
    st.session_state.chat_history.append({"user": query, "bot": answer, "doc": paper.iloc[:,[0,2]], "rating": "평가없음"})
    
    # 현재 질문을 보여줌. -> 평가 받고, stream 옵션 적용 위해
    chat_now = st.session_state.chat_history[::-1][0]
    
    col1, col2, col3 = st.columns([1.2,8,1])
    with col1:
        pass
    with col2:
        st.markdown('')
        st.markdown(f"<div style='background-color: #f9f9f9; padding: 10px;'>{chat_now['user']}</div><br>", unsafe_allow_html=True)
    with col3:
        st.image('images/jpg/궁금-100.png', width = 80)
        
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('images/jpg/안녕-100.png', width = 70)
    with col2:
        st.markdown(f"<div style='background-color: #EBF5FB; padding: 10px;'>{chat_now['bot']}</div><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([7,3,2.5,2.5])
    with col1:
        st.write("이 답변에 대한 평가를 남겨주세요:")
    with col2:
        rat1 = st.checkbox("불만족:thumbsdown:", key=f"rat1_{len(st.session_state.conversation_history)}")
        if rat1:
            update_ratings("불만족")
    with col3:
        rat2 = st.checkbox("보통:smile:", key=f"rat2_{len(st.session_state.conversation_history)}")
        if rat2:
            update_ratings("보통")
    with col4:
        rat3 = st.checkbox("만족:thumbsup:", key=f"rat3_{len(st.session_state.conversation_history)}")
        if rat3:
            update_ratings("만족")

    st.markdown("답변에 참고한 문서입니다.")
    # paper.rename(columns = {'text':'참고 내용','paper_title':'문서 제목'})
    chat_now['doc']
    
# 대화 기록 출력
st.markdown('')
st.markdown('')
st.warning('이전 대화 기록:speech_balloon:')

# st.markdown('--- 이전 대화 기록:speech_balloon: ---')
for chat in st.session_state.chat_history[::-1]:
    col1, col2, col3 = st.columns([1.2,8,1])
    with col1:
        pass
    with col2:
        st.markdown('')
        st.markdown(f"<div style='background-color: #f9f9f9; padding: 10px;'>{chat['user']}</div><br>", unsafe_allow_html=True)
    with col3:
        st.image('images/jpg/궁금-100.png', width = 80)
        
    col1, col2 = st.columns([1,8])
    with col1:
        st.image('images/jpg/안녕-100.png', width = 70)
    with col2:
        st.markdown(f"<div style='background-color: #EBF5FB; padding: 10px;'>{chat['bot']}</div><br>", unsafe_allow_html=True)
    
    chat['rating']
    st.markdown("답변에 참고한 문서입니다.")

    chat['doc']
    
    st.markdown('---')
    