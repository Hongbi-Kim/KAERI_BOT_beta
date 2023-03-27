<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>preprocessing</title>
</head>
<body>
  <h1>🥴PDF 문서 처리 및 텍스트 정제</h1>
  <p>이 코드는 원문자를 일반 숫자로 변환하고, PDF 파일의 텍스트를 분석하여 처리하는 기능을 제공합니다.</p>

  <h2>1️⃣ 주요 함수 및 클래스</h2>
  <ul>
    <li>
      <h3><code>circled_to_normal(char)</code></h3>
      <p>원형 숫자(예: ①)를 일반 숫자로 변환한다.</p>
    </li>
    <li>
      <h3>Chatbot 클래스</h3>
      <p>이 클래스는 PDF 텍스트 분석 및 처리를 수행하는 기능을 제공한다.</p>
      <ul>
        <li>
          <h4><code>parse_paper(pdf)</code></h4>
          <p>주어진 PDF 파일의 텍스트를 분석하여 처리한다.</p>
        </li>
        <li>
            <h4><code>paper_df(pdf)</code></h4>
            <p>분석된 PDF 텍스트를 N글자씩 나눠서 데이터프레임으로 변환한다.</p>
        </li>
    </ul>
    </li>
     </ul>
    <h2>2️⃣ 문제 상황</h2>
    1) 목차 구성이 문서마다 다름.<br>
    <img src="images/문제1.png" alt="문제1" width="70%"><br>
    <img src="images/문제2.png" alt="문제2" width="50%"><br><br>
    2) 띄어쓰기, 부호 등 제대로 안 되어 있음.<br>
    <img src="images/문제3.png" alt="문제3" width="70%"><br><br>
</body>
