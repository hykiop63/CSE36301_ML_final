# Data Preprocessing Plan — Kim Taehoon (Team 5)

## 역할 요약
- **담당**: Overall project management + 데이터 전처리 (TF-IDF 등) + Traditional ML Baselines (Logistic Regression, SVM 등)

---

## 1. 데이터셋 개요

| 항목 | 내용 |
|------|------|
| 이름 | Quora Question Pairs |
| 크기 | 약 404,290개 질문 쌍 |
| 컬럼 | `id`, `qid1`, `qid2`, `question1`, `question2`, `is_duplicate` |
| 레이블 | `is_duplicate` (0 = 비중복, 1 = 중복, 이진 분류) |
| 불균형 | 비중복(0): ~63%, 중복(1): ~37% 클래스 불균형 존재 |
| 결측치 | question1/question2에 1~2개의 null 존재 (매우 적음) |

---

## 2. EDA (탐색적 데이터 분석)

전처리 전 수행할 분석:
- 클래스 분포 시각화 (is_duplicate 비율)
- 질문 길이 분포 (문자 수, 단어 수 기준)
- 단어 겹침 분포 (중복 vs 비중복 간 차이)
- 가장 자주 등장하는 단어 Top N
- 결측치 확인

---

## 3. 기본 텍스트 전처리 (공통)

```python
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                         # 소문자화
    text = re.sub(r'<.*?>', '', text)           # HTML 태그 제거
    text = re.sub(r"what's", "what is", text)   # 축약어 확장
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r'[^\w\s]', '', text)         # 구두점 제거
    text = re.sub(r'\s+', ' ', text).strip()    # 공백 정규화
    return text

# 결측치 처리
df['question1'] = df['question1'].fillna("").apply(clean_text)
df['question2'] = df['question2'].fillna("").apply(clean_text)
```

---

## 4. Feature Engineering

### 4-1. 기본 피처 (Basic Features)

```python
df['q1_len'] = df['question1'].apply(len)           # 문자 길이
df['q2_len'] = df['question2'].apply(len)
df['q1_n_words'] = df['question1'].apply(lambda x: len(x.split()))  # 단어 수
df['q2_n_words'] = df['question2'].apply(lambda x: len(x.split()))

def word_share(row):
    q1 = set(row['question1'].split())
    q2 = set(row['question2'].split())
    if len(q1) + len(q2) == 0:
        return 0
    return len(q1 & q2) / (len(q1) + len(q2))

df['word_share'] = df.apply(word_share, axis=1)     # 단어 겹침 비율
df['abs_len_diff'] = abs(df['q1_len'] - df['q2_len'])
df['mean_len'] = (df['q1_len'] + df['q2_len']) / 2
```

### 4-2. 고급 NLP 피처 (Advanced Features)

```python
stop_words = set(stopwords.words('english'))

def token_features(row):
    q1_tokens = set(row['question1'].split())
    q2_tokens = set(row['question2'].split())
    q1_stops = q1_tokens & stop_words
    q2_stops = q2_tokens & stop_words
    q1_non_stop = q1_tokens - stop_words
    q2_non_stop = q2_tokens - stop_words

    common_tokens = q1_tokens & q2_tokens
    common_stops = q1_stops & q2_stops
    common_non_stop = q1_non_stop & q2_non_stop

    features = {}
    # 공통 토큰 비율 (min/max 기준)
    features['ctc_min'] = len(common_tokens) / max(min(len(q1_tokens), len(q2_tokens)), 1)
    features['ctc_max'] = len(common_tokens) / max(max(len(q1_tokens), len(q2_tokens)), 1)
    # 공통 불용어 비율
    features['csc_min'] = len(common_stops) / max(min(len(q1_stops), len(q2_stops)), 1)
    features['csc_max'] = len(common_stops) / max(max(len(q1_stops), len(q2_stops)), 1)
    # 공통 비불용어 비율
    features['cwc_min'] = len(common_non_stop) / max(min(len(q1_non_stop), len(q2_non_stop)), 1)
    features['cwc_max'] = len(common_non_stop) / max(max(len(q1_non_stop), len(q2_non_stop)), 1)
    # 첫/마지막 단어 일치
    features['first_word_eq'] = int(row['question1'].split()[0] == row['question2'].split()[0]) if q1_tokens and q2_tokens else 0
    features['last_word_eq']  = int(row['question1'].split()[-1] == row['question2'].split()[-1]) if q1_tokens and q2_tokens else 0
    return pd.Series(features)
```

### 4-3. Fuzzy Matching 피처

```python
from fuzzywuzzy import fuzz

df['fuzz_ratio']         = df.apply(lambda r: fuzz.ratio(r['question1'], r['question2']), axis=1)
df['fuzz_partial_ratio'] = df.apply(lambda r: fuzz.partial_ratio(r['question1'], r['question2']), axis=1)
df['token_sort_ratio']   = df.apply(lambda r: fuzz.token_sort_ratio(r['question1'], r['question2']), axis=1)
df['token_set_ratio']    = df.apply(lambda r: fuzz.token_set_ratio(r['question1'], r['question2']), axis=1)
```

### 4-4. TF-IDF 피처

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# 방법 1: q1+q2 concatenation → 개별 벡터화 → cosine similarity
tfidf = TfidfVectorizer(
    max_features=50000,       # 5K~50K 범위에서 튜닝
    stop_words='english',
    ngram_range=(1, 2),       # unigram + bigram
    sublinear_tf=True
)

# 전체 말뭉치로 fit
corpus = pd.concat([df['question1'], df['question2']])
tfidf.fit(corpus)

q1_tfidf = tfidf.transform(df['question1'])
q2_tfidf = tfidf.transform(df['question2'])

# 코사인 유사도 피처
df['tfidf_cosine_sim'] = [
    cosine_similarity(q1_tfidf[i], q2_tfidf[i])[0][0]
    for i in range(len(df))
]

# 방법 2: ML 모델 입력용 sparse matrix 결합
X_tfidf = sp.hstack([q1_tfidf, q2_tfidf])  # (N, 2*max_features)
```

---

## 5. Train/Val/Test 분할

```python
from sklearn.model_selection import train_test_split

X = feature_matrix   # 수동 피처 + TF-IDF sparse matrix
y = df['is_duplicate']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# 최종: 80% train / 10% val / 10% test
```

---

## 6. 베이스라인별 전처리 방법 비교

### Baseline 1: anokas — Data Analysis & XGBoost Starter
- **텍스트 정제**: 소문자화, 축약어 확장, 구두점 제거
- **결측치**: `question1/2.fillna("")`로 빈 문자열 치환
- **주요 피처**:
  - `word_share` (핵심 피처 — 중복/비중복 구분력 매우 높음)
  - `q1_n_words`, `q2_n_words` (단어 수)
  - `freq_qid1`, `freq_qid2` (질문 ID 빈도 — 리크 피처, 사용 주의)
- **벡터화**: TF-IDF 없이 수동 피처만 사용
- **모델**: XGBoost
- **성능**: LB 0.35460 (log loss 기준, 낮을수록 좋음)

### Baseline 2: nicapotato — TF-IDF + Logistic Regression
- **텍스트 정제**: 소문자화, 불용어 제거, 구두점 제거
- **결측치**: null 제거
- **주요 피처**:
  - TF-IDF 벡터 (max_features 미정, stop_words='english')
  - 기본 길이/단어 수 피처
  - 코사인 유사도
- **벡터화**: `TfidfVectorizer` → q1, q2 각각 변환 후 결합
- **모델**: Logistic Regression
- **특징**: 가장 간단한 구조, 빠른 학습/추론 → 우리 프로젝트의 핵심 baseline

### Baseline 3: vabatista — BERT Transfer Learning
- **텍스트 정제**: 최소화 (BERT tokenizer가 자체 처리)
- **결측치**: null 제거
- **토크나이저**: `BertTokenizer.from_pretrained('bert-base-uncased')`
- **인코딩**:
  ```
  [CLS] question1 [SEP] question2 [SEP]
  ```
  - `max_length=128`, padding='max_length', truncation=True
  - attention_mask 생성
- **모델**: BERT Cross-Encoder (Fine-tuning)
- **특징**: 전처리 최소 → 모델이 모든 처리 담당, 가장 높은 정확도

---

## 7. 우리 프로젝트 전처리 실행 계획

| 단계 | 작업 | 담당 |
|------|------|------|
| 1 | EDA: 클래스 분포, 길이 분포, 단어 겹침 분포 시각화 | Taehoon |
| 2 | 결측치 처리 (fillna + 로그 기록) | Taehoon |
| 3 | 텍스트 클리닝 함수 구현 (축약어 확장 포함) | Taehoon |
| 4 | Basic Features 생성 (word_share, 길이, 단어 수) | Taehoon |
| 5 | Advanced NLP Features (ctc, csc, cwc, fuzzy) | Taehoon |
| 6 | TF-IDF 벡터화 (5K~50K 범위 실험) | Taehoon |
| 7 | 코사인 유사도 피처 생성 | Taehoon |
| 8 | Train/Val/Test 분할 (80/10/10) | Taehoon |
| 9 | Logistic Regression, SVM, NB, DT, RF, GBM 학습 | Taehoon |
| 10 | 피처 수(TF-IDF dim), 정규화(L1/L2), 트리 depth 튜닝 | Taehoon |

---

## 8. 최종 피처 구성 (예상)

| 피처 그룹 | 수 | 설명 |
|-----------|----|------|
| 기본 피처 | ~8 | 길이, 단어 수, word_share |
| 고급 NLP 피처 | ~13 | ctc, csc, cwc, fuzzy 등 |
| TF-IDF 코사인 유사도 | 1 | q1-q2 간 유사도 |
| TF-IDF sparse | 2×N | ML 모델 직접 입력용 |

---

## 참고 자료

- [anokas XGBoost Starter](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb)
- [nicapotato TF-IDF + LR](https://www.kaggle.com/code/nicapotato/tf-idf-and-features-logistic-regression)
- [vabatista BERT](https://www.kaggle.com/code/vabatista/quora-question-pairs-transfer-learning-w-bert)
- [Vedansh Sharma — Analytics Vidhya](https://medium.com/analytics-vidhya/quora-question-pairs-similarity-problem-8e3ae90441f0)
- [Shritam Mund — Medium](https://shritam.medium.com/quora-question-pair-similarity-case-study-54f0d8c9b630)
