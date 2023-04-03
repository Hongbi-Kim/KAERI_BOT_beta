import streamlit as st
import requests
import pandas as pd
import numpy as np
import openai, uuid, time
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import os, re, tenacity, pickle
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account

openai.api_key = st.secrets['openai']
# 서비스 계정 키 파일로부터 인증 정보를 가져옵니다.
credentials = service_account.Credentials.from_service_account_file(st.secrets['translate'])

# 인증 정보를 사용하여 클라이언트를 생성합니다.
translate_client = translate.Client(credentials=credentials)


col1, col2 = st.columns([1,8])
with col1:
    st.image('images/jpg/mc파동이-100.png', width = 100)
with col2:
    st.title('파동이봇')
st.markdown('')

with open('정제파일/원예규_note_조.pkl', 'rb') as f:
    df_500 = pickle.load(f)
with open('원규_note_up150_300.pkl', 'rb') as f:
    df_300 = pickle.load(f)

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
def search_embeddings(df, query, n=3, pprint=True):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    sources = []
    for i in range(3):
        sources.append({results.iloc[i,0][:150]+'...'})
    print(sources)
    return results.head(3)

def create_prompt(df, query, conversation):
    result = search_embeddings(df, query)
    
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

def create_prompt_en(df, query, conversation):
    result = search_embeddings(df, query)
    
    system_role = f"""You are an AI language model named "파동이" whose expertise is reading and summarizing Korea Atomic Energy Research Institute regulatory documents. 
    You must take the given embeddings and return a very detailed summary of the document in the language of the query. 
    Your conversation history is as follows:

        {conversation}
                
        Here are the embeddings: 
                
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """
        
        You must return in English. Please answer using polite and formal.
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

def translate_text(text, target_language='ko'):
    result = translate_client.translate(text, target_language)
    return result['translatedText']

if 'chat_history' not in st.session_state:
    st.session_state.conversation_history = []
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

conversation = ""

# 이미지와 text_input을 함께 출력합니다.
col1, col2 = st.columns([1,8])
with col1:
    st.image('images/jpg/궁금-100.png', width = 90)
with col2:
    query = st.text_input("규정에 관련된 질문을 해주세요.:sunglasses: 새로운 주제로 넘어갈 때, [새로운 주제] 버튼을 누른 뒤 질문을 해주세요.", "", placeholder="예시: 휴가 결재는 누구한테 받아야 돼?")
    
col1, col2, col3 = st.columns([6,2,1])
with col1:
    pass
with col2:
    if st.button("새로운 주제", key='new1'): 
        st.session_state.conversation_history = []
        st.session_state.chat_history = []
        conversation = ""
with col3:
    if st.button("SEND", key='message3') or len(query) > 1:
        st.session_state.conversation_history.append(query)
        if st.session_state.conversation_history:
            for i, chat in enumerate(st.session_state.conversation_history[:-1]):
                conversation += f"User: {chat}\n"
                if i < len(st.session_state.conversation_history) - 2:
                    conversation += f"Assistant: {st.session_state.chat_history[i]['bot']}\n"
            conversation += f"User: {st.session_state.conversation_history[-1]}\n"
        prompt = create_prompt(df_500, query, st.session_state.conversation_history)
        response = gpt(prompt)
        answer = response['answer']
        print(answer)
        paper = search_embeddings(df_500, query)
        st.session_state.chat_history.append({"user": query, "bot": answer, "doc": paper.iloc[:,[0,2]]})

# 대화 기록 출력
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
    
    st.markdown("답변에 참고한 문서입니다.")
    # paper.rename(columns = {'text':'참고 내용','paper_title':'문서 제목'})
    chat['doc']
    st.markdown('---')
