# Quora Question Pairs — 데이터 & 전처리 요약 / Data & Preprocessing Summary

> Team 5 | Taehun Kim — `01_preprocessing.ipynb`

---

# 🇰🇷 한글 버전

## 1. 데이터 성질 (Data Characteristics)

| 항목 | 값 |
|------|-----|
| 총 질문 쌍 | **404,290건** |
| 컬럼 | id, qid1, qid2, question1, question2, is_duplicate |
| 결측치 | question1: 1건, question2: 2건 → **빈 문자열로 대체** |
| 클래스 분포 | 중복 36.9% / 비중복 63.1% (**불균형**) |
| 질문 길이 | 평균 ~11단어, 대부분 60단어 이내 |

**핵심 인사이트 (EDA)**
- **클래스 불균형** — 비중복이 약 1.7배 많음 → accuracy만으로 판단 위험 (F1 필요)
- **Word Share Ratio**(단어 겹침 비율)가 중복/비중복을 가장 잘 구분하는 신호 → 수동 피처의 핵심 근거
- 길이 분포는 두 질문이 유사 → 길이 차이도 보조 신호

## 2. 공통 전처리 (Common Preprocessing — 모든 파트 공유)

> 모든 팀원이 동일하게 사용하는 기반 단계. 여기서 데이터 정합성과 재현성을 보장함.

1. **데이터 로딩 & 결측치 처리**
   - `train.csv` 로드 (404,290행)
   - question1/question2 결측치(총 3건) → **빈 문자열(`''`)로 대체**

2. **Train / Val / Test Split (전 모델 공통)**
   - **80 / 10 / 10** 비율 (train 323,432 / val 40,429 / test 40,429)
   - `stratify=y` → 클래스 비율 유지, `random_state=42` 고정
   - 분할 **인덱스를 `splits/`에 저장** → git 공유로 **팀 전원 동일 split 보장**
   - 효과: 각 팀원이 raw CSV 로드 후 같은 인덱스로 슬라이싱 → random_state 일치 증명 불필요

3. **Data Leakage 방지 원칙**
   - TF-IDF / 단어사전(vocab)은 **train만으로 fit** → val/test 정보 누수 차단

## 3. 전처리 전략 — "모델별 맞춤 클리닝"

> 한 번 로딩 후 **3개 파이프라인**으로 분기. 모델 특성에 맞춰 클리닝 강도를 다르게 적용한 것이 핵심.

| 파이프라인 | 대상 모델 | 클리닝 방식 | 근거 |
|-----------|----------|------------|------|
| **for_ml** | 전통 ML | 소문자화 + 축약어 확장 + 구두점 제거, **숫자 유지** | anokas/nicapotato 베이스라인 |
| **for_lstm** | Siamese LSTM | 위와 동일 + **숫자 제거** | 임베딩 vocab에 숫자 적음 |
| **for_bert** | BERT | **최소 클리닝**(HTML·공백만) | WordPiece가 대소문자·구두점 자체 처리 |

**왜 BERT는 클리닝을 안 하나?** → 소문자화·구두점 제거 시 오히려 성능 저하 (vabatista 근거)

## 4. 모델별 산출물 (Outputs)

**① for_ml (전통 ML)**
- **수동 피처 10개**: 길이, 단어 수, 공통 단어, `word_share`(|A∩B|/(|A|+|B|)), `tfidf_cosine`
- **TF-IDF**: max 50K, unigram, 불용어 제거, min_df=2 → (323432, **93012**) sparse matrix

**② for_lstm**
- 단어사전 **44,820개**(빈도 2↑), `<PAD>`/`<UNK>` 토큰
- 시퀀스 패딩 길이 50 → (N, 50) npy

**③ for_bert**
- 최소 클리닝된 텍스트 CSV (question1, question2, label)

## 5. 발표 강조 포인트
1. **클래스 불균형 + word_share가 핵심 신호** (데이터 성질)
2. **공통 split을 git으로 공유**해 팀 전체 결과 비교 가능 (협업·재현성 설계)
3. **모델별 맞춤 전처리** — 같은 데이터, 다른 클리닝 강도 (전처리 차별점)

---
<br>

# 🇺🇸 English Version

## 1. Data Characteristics

| Item | Value |
|------|-------|
| Total question pairs | **404,290** |
| Columns | id, qid1, qid2, question1, question2, is_duplicate |
| Missing values | question1: 1, question2: 2 → **filled with empty string** |
| Class distribution | Duplicate 36.9% / Non-duplicate 63.1% (**imbalanced**) |
| Question length | ~11 words on average, mostly within 60 words |

**Key Insights (EDA)**
- **Class imbalance** — non-duplicates are ~1.7× more frequent → accuracy alone is misleading (use F1)
- **Word Share Ratio** is the strongest separating signal between duplicate / non-duplicate → main rationale for hand-crafted features
- Length distributions of the two questions are similar → length difference is a supporting signal

## 2. Common Preprocessing (Shared Across All Parts)

> The shared foundation every team member uses. Ensures data consistency and reproducibility.

1. **Data loading & missing-value handling**
   - Load `train.csv` (404,290 rows)
   - Missing question1/question2 (3 total) → **replaced with empty string (`''`)**

2. **Train / Val / Test split (shared by all models)**
   - **80 / 10 / 10** ratio (train 323,432 / val 40,429 / test 40,429)
   - `stratify=y` to preserve class ratio, fixed `random_state=42`
   - **Split indices saved to `splits/`** → shared via git to **guarantee identical splits for the whole team**
   - Benefit: each member loads the raw CSV and slices by the same indices → no need to prove random_state matches

3. **Data-leakage prevention**
   - TF-IDF / vocabulary are **fit on train only** → blocks val/test information leakage

## 3. Preprocessing Strategy — "Model-Specific Cleaning"

> Load once, then branch into **3 pipelines**. The core idea: vary cleaning intensity to fit each model.

| Pipeline | Target Model | Cleaning | Rationale |
|----------|-------------|----------|-----------|
| **for_ml** | Traditional ML | lowercase + expand contractions + remove punctuation, **keep numbers** | anokas/nicapotato baseline |
| **for_lstm** | Siamese LSTM | same as above + **remove numbers** | few numbers in embedding vocab |
| **for_bert** | BERT | **minimal cleaning** (HTML & whitespace only) | WordPiece handles casing/punctuation itself |

**Why no cleaning for BERT?** → Lowercasing / removing punctuation can actually hurt performance (vabatista rationale)

## 4. Outputs per Model

**① for_ml (Traditional ML)**
- **10 hand-crafted features**: lengths, word counts, common words, `word_share` (|A∩B|/(|A|+|B|)), `tfidf_cosine`
- **TF-IDF**: max 50K, unigram, stopwords removed, min_df=2 → (323432, **93012**) sparse matrix

**② for_lstm**
- Vocabulary of **44,820** tokens (freq ≥ 2), `<PAD>`/`<UNK>` tokens
- Padded sequences of length 50 → (N, 50) npy

**③ for_bert**
- Minimally cleaned text CSV (question1, question2, label)

## 5. Presentation Highlights
1. **Class imbalance + word_share as the key signal** (data characteristics)
2. **Common split shared via git** enables fair team-wide comparison (collaboration & reproducibility)
3. **Model-specific preprocessing** — same data, different cleaning intensity (preprocessing differentiator)

---
<br>

# 📊 PPT용 슬라이드 문구 / PPT-Ready Slide Text

> 발표에 바로 붙여넣을 수 있는 슬라이드 단위 문구 모음.

## 📑 슬라이드 1 — 데이터셋 & EDA (한글)

**▎데이터 개요**
- Quora 질문 쌍 데이터셋 — 총 **404,290 쌍**, 결측치 **3건**은 빈 문자열로 처리

**▎클래스 분포**
- 중복 **36.9%** vs 비중복 **63.1%** → **클래스 불균형**
- → 정확도(accuracy)만으로는 부족, **F1-score 기준 평가**

**▎질문 길이**
- 질문 하나당 평균 약 **11단어**, 대부분 **20단어 이내**의 짧은 문장

**▎단어 중복도 (Word Share)**
- 두 질문의 **단어 겹침 비율**이 중복/비중복을 가장 잘 구분
- → 수동 피처 설계의 **핵심 근거**

## 📑 Slide 1 — Dataset & EDA (English)

**▎Overview**
- Quora question-pair dataset — **404,290 pairs**, with only **3 missing values** filled as empty strings

**▎Class Distribution**
- Duplicate **36.9%** vs Non-duplicate **63.1%** → **class imbalance**
- → accuracy alone is insufficient; evaluate with **F1-score**

**▎Question Length**
- Each question averages about **11 words**, mostly short (within ~20 words)

**▎Word Share**
- The **word-overlap ratio** between the two questions best separates duplicates from non-duplicates
- → key rationale for our **hand-crafted features**

---

## 📑 슬라이드 2 — 전처리 (한글)

**▎모델별 맞춤 클리닝**
- 한 번 로딩 후 **3개 파이프라인**으로 분기 — 모델 특성에 맞게 클리닝 강도 차등
- **전통 ML / LSTM**: 소문자화 + 축약어 확장 + 구두점 제거 (정제)
- **BERT**: **최소 처리**(HTML·공백만) — WordPiece가 대소문자·구두점 자체 처리

**▎텍스트 클리닝 3단계 (ML / LSTM)**
- **① 소문자화**: `What IS` → `what is` (단어 통일·일관성)
- **② 축약어 확장**: `what's` → `what is` (단어 중복도 정확도↑)
- **③ 구두점 제거**: `(Koh-i-Noor)?` → `koh i noor` (TF-IDF 노이즈 감소)
- 숫자는 ML에서 **유지**(예: "iphone 14"), LSTM에선 **제거**(임베딩 vocab에 숫자 적음)

**▎공통 처리 & 재현성**
- **Train / Val / Test = 80 / 10 / 10**, stratify로 클래스 비율 유지
- split 인덱스를 **git 공유** → 팀 전체가 동일 데이터로 비교
- TF-IDF·단어사전은 **train만으로 학습** → 데이터 누수(leakage) 방지

## 📑 Slide 2 — Preprocessing (English)

**▎Model-Specific Cleaning**
- Load once, then branch into **3 pipelines** — cleaning intensity tuned per model
- **Traditional ML / LSTM**: lowercase + expand contractions + remove punctuation
- **BERT**: **minimal cleaning** (HTML & whitespace only) — WordPiece handles casing/punctuation

**▎3-Step Text Cleaning (ML / LSTM)**
- **① Lowercasing**: `What IS` → `what is` (unify word forms)
- **② Expand contractions**: `what's` → `what is` (improves word-share accuracy)
- **③ Remove punctuation**: `(Koh-i-Noor)?` → `koh i noor` (reduces TF-IDF noise)
- Numbers **kept** for ML (e.g. "iphone 14"), **removed** for LSTM (few in embedding vocab)

**▎Common Processing & Reproducibility**
- **Train / Val / Test = 80 / 10 / 10**, stratified to keep class ratio
- Split indices **shared via git** → whole team compares on identical data
- TF-IDF & vocabulary **fit on train only** → prevents data leakage

---

## 📑 슬라이드 3 — 전통적 ML 베이스라인 (한글)

**▎실험 구성**
- **8개 모델 × 2가지 피처** 비교 — 수동 피처(11개) vs TF-IDF
- 모델: LR · SVM · Naive Bayes · Decision Tree · Random Forest · Gradient Boosting · **XGBoost**
- Validation Log Loss 기준 하이퍼파라미터 튜닝

**▎핵심 결과 (대표 모델)**

| 모델 | 피처 | Test Acc | Test F1 | 추론속도 |
|------|------|----------|---------|---------|
| LR | TF-IDF | 75.6% | 0.628 | 빠름 |
| **XGBoost** | 수동 피처 | **73.2%** | **0.651** | 빠름 |
| Random Forest | 수동 피처 | 72.8% | 0.647 | 느림 |

- **최고 성능: XGBoost** — F1 **0.651**, Log Loss **0.479**(최저)
- `tfidf_word_share`(단어 중복도)가 가장 강력한 피처

**▎한계 → 딥러닝의 필요성**
- 정확도 **73~76%에서 정체** → 단어 겹침 기반이라 **문맥·의미를 못 잡음**
- 예: "How to learn Python" vs "Best way to study Python" → 같은 뜻이지만 단어가 다르면 놓침
- → **문맥 이해(Word2Vec + LSTM)로 확장 동기**

## 📑 Slide 3 — Traditional ML Baselines (English)

**▎Setup**
- Compared **8 models × 2 feature types** — hand-crafted (11) vs TF-IDF
- Models: LR · SVM · Naive Bayes · Decision Tree · Random Forest · Gradient Boosting · **XGBoost**
- Hyperparameters tuned on validation log loss

**▎Key Results**

| Model | Features | Test Acc | Test F1 | Inference |
|-------|----------|----------|---------|-----------|
| LR | TF-IDF | 75.6% | 0.628 | Fast |
| **XGBoost** | Hand-crafted | **73.2%** | **0.651** | Fast |
| Random Forest | Hand-crafted | 72.8% | 0.647 | Slow |

- **Best: XGBoost** — F1 **0.651**, lowest Log Loss **0.479**
- `tfidf_word_share` (word-overlap) was the strongest feature

**▎Limitation → Motivation for Deep Learning**
- Accuracy **plateaus at 73–76%** → relies on word overlap, **misses context/meaning**
- e.g. "How to learn Python" vs "Best way to study Python" — same intent, different words → missed
- → motivates **contextual understanding (Word2Vec + LSTM)**

---
<br>

# 🗒️ 발표 대비 메모 (대본 작성용 / Q&A)

> 슬라이드에는 안 넣지만, 대본 쓸 때·질문 대응용으로 기억해둘 포인트.

**▎`tfidf_word_share`는 어디서 생겼나? (피처 11개의 출처)**
- 01 전처리에서 만든 수동 피처는 **10개** (`word_share`, `tfidf_cosine` 포함, `tfidf_word_share`는 없음)
- `tfidf_word_share`는 **02 모델링 단계에서 추가**된 11번째 파생 피처
- 계산: raw 텍스트 없이 01이 저장한 **TF-IDF sparse matrix만으로** 직접 산출
  - 수식: Σ idf(공통단어) / Σ idf(Q1∪Q2) — `word_share`에 **희귀 단어 IDF 가중치**를 더한 버전 (anokas 베이스라인 핵심 피처)
- ⚠️ 주의: **전처리(클리닝/split)는 01과 동일**, 바뀐 것 없음. 단지 "피처가 만들어진 위치"가 02일 뿐.
- 예상 질문 대응: "피처 10개라며 왜 11개?" → "01의 10개 + 02에서 추가한 tfidf_word_share = 11개"
