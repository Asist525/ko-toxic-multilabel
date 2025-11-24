````markdown
# 한국어 욕설 탐지 프로젝트

한국어 문장을 욕설/비욕설로 이진 분류하는 프로젝트이다.  
char 단위 TF-IDF로 벡터화하고 LinearSVC로 분류한다.  
검증 단계에서 틀린 샘플을 CSV로 저장해 다음 학습 라운드에 재사용할 수 있도록 한다.  
추론 단계에서는 임계값(threshold)을 바꿔 욕설 판정 강도를 조절할 수 있다.

---

## 1. 폴더 구조

```text
project-root/
├─ dataset/
│  ├─ hate_speech_data.csv
│  └─ hate_speech_binary_dataset2.csv
├─ models/
│  ├─ vec_char_1_3_500k.joblib
│  └─ clf_linearsvc_v1.joblib
├─ logs/
│  ├─ hard_cases_round1.csv
│  ├─ hard_missed_label1_round1.csv
│  └─ hard_false_label1_round1.csv
├─ train.py
├─ predict.py
└─ README.md
````

* `dataset/` : 학습에 사용할 CSV/TSV 파일을 넣는 디렉터리
* `models/` : 학습이 끝난 벡터라이저와 분류 모델이 저장되는 디렉터리
* `logs/` : 검증에서 오분류된 샘플을 라운드별로 저장하는 디렉터리
* `train.py` : 학습 및 로그 생성 스크립트
* `predict.py` : 저장된 모델로 예측하는 스크립트

---

## 2. 데이터 형식

기본 CSV 형식:

* `문장` : 텍스트
* `혐오 여부` : 0 또는 1

라벨 의미는 다음과 같이 통일한다.

* `혐오 여부 = 0` → 욕설/혐오
* `혐오 여부 = 1` → 정상/비욕설

TSV 파일을 사용할 때 컬럼 이름이 다르면 학습 전에 `문장`, `혐오 여부`로 리네임해서 사용한다.
길이가 500자를 초과하는 문장은 제거한다.
`문장` 또는 `혐오 여부`가 결측인 행은 제거한다.

---

## 3. 실행 환경

```bash
pip install pandas scikit-learn python-pptx
```

(PPT 생성이 필요 없으면 `python-pptx`는 생략할 수 있다.)

---

## 4. 학습 (train.py)

```bash
python train.py
```

처리 순서:

1. `./dataset/` 아래의 CSV/TSV를 읽어서 하나로 합친다.
2. 컬럼을 `문장`, `혐오 여부`로 맞추고 결측치와 500자 초과 문장을 제거한다.
3. train/valid로 분할한다.
4. `TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=500000)` 으로 벡터화한다.
5. `LinearSVC(class_weight={0: 1.0, 1: 1.2})` 로 학습한다.
6. 검증셋을 예측해 오분류 샘플을 `./logs/`에 CSV로 저장한다.
7. 학습된 벡터라이저와 모델을 `./models/`에 저장한다.

로그 파일 종류:

* `hard_cases_round{n}.csv` : 그 라운드에서 틀린 샘플 전체
* `hard_missed_label1_round{n}.csv` : 원래 1인데 0으로 예측한 샘플
* `hard_false_label1_round{n}.csv` : 원래 0인데 1로 예측한 샘플

이 로그들을 다음 라운드 학습에 다시 포함시키면 점진적으로 개선할 수 있다.

---

## 5. 추론 (predict.py)

```bash
python predict.py
```

처리 순서:

1. `./models/vec_char_1_3_500k.joblib`
2. `./models/clf_linearsvc_v1.joblib`

을 불러온다.

코드 내에 정의된 예시 문장을 벡터화 → 예측하고, 출력 시 다음과 같이 두 그룹으로 나눠 보여준다.

```text
=== 욕설 판정 ===
- ...

=== 비욕설 판정 ===
- ...
```

모델은 내부적으로 “0=욕설, 1=비욕설”로 학습되어 있으므로, 예측 후 사람이 보기 좋게 라벨을 다시 이름 붙여서 출력한다.

---

## 6. 강도(임계값) 조절

`predict.py`의 핵심 함수는 SVM의 `decision_function` 값을 사용한다.
기본은 `threshold = 0.0` 이며, 다음과 같이 동작한다.

```python
scores = clf.decision_function(X)
orig_labels = (scores >= threshold).astype(int)   # 1=비욕설, 0=욕설
hate_labels = (orig_labels == 0).astype(int)      # 1=욕설, 0=비욕설
```

* `threshold` 를 **낮추면** (예: `-0.8`) → 아주 강한 욕설만 욕설로 본다.
* `threshold` 를 **올리면** (예: `0.3`) → 애매한 표현도 욕설로 본다.

서비스에서 욕설 차단 강도를 설정할 때 이 값을 조절해서 사용한다.

---

## 7. 모델 구성 요소

* 벡터라이저: char 단위 TF-IDF

  * 오타, 띄어쓰기 변형, 특수문자가 섞인 한국어 욕설을 처리하기 위한 선택
* 분류기: LinearSVC

  * 희소 행렬 처리에 적합하고, 텍스트 분류에서 성능이 안정적임

---

## 8. 노트북

* `v1.ipynb` : 데이터 분포 확인, 1차 전처리, char TF-IDF 실험
* `v2.ipynb` : max_features, n-gram 범위, 오분류 샘플 탐색, 하드케이스 추출

두 노트북은 실험 기록이며 제출/실행 대상은 `train.py`, `predict.py` 이다.

```
```
