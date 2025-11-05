# 한국어 욕설 탐지 프로젝트

한국어 문장을 **욕설인지 아닌지** 이진으로 분류하는 프로젝트다.  
기본 모델은 **char 단위 TF-IDF + LinearSVC** 조합을 사용하고, 학습할 때마다 **틀린 샘플만 모아서 다시 학습에 넣는(hard-case 라운드)** 구조로 만들어져 있다.  
또한 예측 시 **임계값(threshold)** 을 바꿔서 “강한 욕만 잡기 / 애매한 것도 잡기”를 조절할 수 있다.

---

## 1. 디렉터리 구조

```text
project-root/
├─ dataset/
│  ├─ hate_speech_data.csv
│  └─ hate_speech_binary_dataset2.csv
├─ models/
│  ├─ vec_char_1_3_500k.joblib     ← 학습 후 생성
│  └─ clf_linearsvc_v1.joblib      ← 학습 후 생성
├─ logs/
│  ├─ hard_cases_round1.csv
│  ├─ hard_missed_label1_round1.csv
│  └─ hard_false_label1_round1.csv
├─ train.py
├─ predict.py
└─ README.md
