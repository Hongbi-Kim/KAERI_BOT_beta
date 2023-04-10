<h1>
    <img src="images/안녕-100.png" alt="파동이" width="60" height="60"> 파동이봇
</h1>
<br>
<h2>1️⃣ 파동이봇</h2>
파동이봇은 규정에 관한 질문에 대해 자세하고 정확한 답변을 제공하는 앱입니다.<br>
이 앱은 Python으로 작성되었으며, GPT-3.5-turbo 기반의 OpenAI API와 Streamlit을 활용하여 구현되었습니다.<br>
사용자로부터 입력된 질문을 바탕으로, 한국원자력연구원의 규정 문서에서 유사한 내용을 찾아 대답합니다.<br>
보안의 이유로 본 REPOSITORY에는 일부 파일만 공개됩니다.<br>
<br>
<h3> 👇🏻 예시 </h3>
1) 초기화면<br>
<img src="images/예시3.JPG" alt="앱1" width="70%"><br>

2) 질문 입력<br>
<img src="images/410_예시4.JPG" alt="앱1" width="70%"><br>

3) 답변 생성 및 참고 문서 제공<br>
<img src="images/410_예시3.JPG" alt="앱1" width="70%">
<br>
<br>
<h2>2️⃣ 주요 기능</h2>
&nbsp; - 사용자의 질문에 대한 정확하고 자세한 답변 제공<br>
&nbsp; - OpenAI API를 이용한 텍스트 임베딩 및 문장 간 유사성 계산<br>
&nbsp; - 대화 기록을 관리하고, 사용자와 파동이봇 간의 소통을 시각화<br>
<br>

<h2>3️⃣ 앱 구성</h2>
&nbsp; - 질문 입력: 사용자로부터 규정에 관한 질문을 입력받음<br>
&nbsp; - 답변 생성: 입력된 질문을 바탕으로, 한국원자력연구원의 규정 문서에서 유사한 내용을 찾아 답변 생성<br>
&nbsp; - 대화 기록: 사용자와 파동이봇 간의 소통 내용을 시각화하고 관리<br>
&nbsp; - 참고 문서: 답변에 참고한 문서를 사용자에게 제공.<br>
<br>

<h2>4️⃣ 개발 도구</h2>
<div>
<img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?&style=for-the-badge&logo=Streamlit&logoColor=white"/>
<img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-412991.svg?&style=for-the-badge&logo=OpenAI&logoColor=white"/><br>
</div>
<br>
<h2>5️⃣ 주요 개선 사항</h2>
<table>
  <thead>
    <tr>
      <th>개선 날짜</th>
      <th>모델</th>
      <th>서비스</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td> 2023-03-24<br><br><br></td>
      <td>
        1. 이전 질문을 고려하여 답변 생성(추가 개선 중)<br>
        2. 답변 말투 일관되게 수정<br><br></td>
      <td>
        1. 이전 질문-답변을 페이지에 유지<br>
        2. 예시 질문 추가<br>
        3. 파동이 그림 추가 및 투명 배경으로 수정<br></td>
    </tr>
    <td> 2023-03-26<br><br></td>
      <td>
        1. 속도 개선을 위한 시도 → 빨라짐. 하지만 번역 오류<br>
        한국어 질문 → 영어로 gpt에 입력 → 영어로 된 답변을 한국어로 번역</td>
      <td>
        <br><br></td>
    </tr>
    <td> 2023-03-27<br><br></td>
      <td>
        1. 전처리 꼼꼼하게 다시!!<br>
        [비교]<br>
        2. 문서 단락 300/ 500글자씩 나눈 뒤 비교<br>
        3. 영어 번역 vs. 300 vs. 500</td>
      <td>
        [비교]<br>
        1. 더 빠른 답변(번역)/ 빠른 답변(300)/ 정확한 답변(500) 옵션 추가 (For test)<br>
        2. 새로운 주제 옵션 추가 - 관련 없는 이전 질문에 영향을 받기 때문이다…</td>
    </tr>
    </tr>
    <td>2023-04-03<br><br></td>
      <td>
        1. 전처리 꼼꼼하게 문서 다 살펴보기😵‍💫 <br>
        <li>맞춤법, 기호</li>
        <li>제*장, 제*조로 나눠서 내용의 흐름이 끊기지 않게 수정</li>
        <a href="https://github.com/Hongbi-Kim/KAERI_BOT_beta/blob/master/preprocessing">👉🏻 전처리 code 및 설명</a> 
      <td>
        <br>
      </td>
    </tr>
    <td> 2023-04-07<br><br></td>
      <td>
        1. 질문 관련 문서 검색 process 변경<br>
        <li>cosine similarity : 2개 추출</li>
        <li>bm25 : 1개 추출</li>
        2. 문서 추가 및 문서를 더 세분화해서 전처리
      <td>
        1. DB 저장(질문-답변-평가)<br>
        2. 답변 만족도 체크 기능<br>
        3. gpt 답변 생성 서버와 streamlit 앱 서버 분리<br>(분산시스템 구축)</td>
    </tr>
    </tr>
  </tbody>
</table>
<br>
<h2>6️⃣ 참고 자료</h2>
&nbsp; - <a href="https://github.com/mukulpatnaik/researchgpt">Researchgpt</a>
<p>&nbsp; - <a href="https://cloud.google.com/sdk/docs?hl=ko">Google Cloud</a></p>