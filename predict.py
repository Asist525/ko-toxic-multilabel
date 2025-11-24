# predict.py
from __future__ import annotations
from pathlib import Path
from typing import List, Union, Dict, Any

import joblib
import numpy as np

# ================================
# 경로 / 설정
# ================================
# 노트북/스크립트 모두 동작하도록 처리
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

MODEL_DIR = BASE_DIR / "models"

VEC_PATH = MODEL_DIR / "vec_char_1_3_300k.joblib"
CLF_PATH = MODEL_DIR / "clf_linearsvc_c05_bal.joblib"

# 최종 결론: LinearSVC margin(score) >= 0 이면 욕설(1)
DEFAULT_THRESHOLD = 1.0 # Defaut는 0이나 강한 욕설만을 걸러내거나, 확실한 욕설을 걸러낼 경우 1 사용


# ================================
# 모델 로드
# ================================
def load_model():
    """
    학습된 TF-IDF 벡터라이저와 LinearSVC 모델을 로드한다.
    """
    if not VEC_PATH.exists():
        raise FileNotFoundError(f"벡터라이저 파일을 찾을 수 없습니다: {VEC_PATH}")
    if not CLF_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {CLF_PATH}")

    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    return vec, clf


# ================================
# 예측 함수
# ================================
def predict_hate(
    texts: Union[str, List[str]],
    hate_threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    욕설/비욕설 예측 함수.

    - 입력:
        texts: 문자열 하나 또는 문자열 리스트
        hate_threshold:
           LinearSVC의 decision_function 스코어 기준으로
           score >= hate_threshold 이면 욕설(1)로 본다.
           기본값은 0.0 (margin 0 기준).

    - 출력:
        [
          {
            "text": 원문 문자열,
            "score": float (decision_function 점수, 양수일수록 욕설 쪽),
            "hate_label": int (1=욕설, 0=비욕설),
            "hate_str": "욕설" 또는 "비욕설",
          },
          ...
        ]
    """
    if isinstance(texts, str):
        texts = [texts]

    vec, clf = load_model()

    X = vec.transform(texts)
    scores = clf.decision_function(X)  # shape (n_samples,)

    results: List[Dict[str, Any]] = []
    for t, s in zip(texts, scores):
        hate_label = 1 if s >= hate_threshold else 0
        results.append(
            {
                "text": t,
                "score": float(s),
                "hate_label": int(hate_label),
                "hate_str": "욕설" if hate_label == 1 else "비욕설",
            }
        )
    return results


# ================================
# 간단 테스트
# ================================
if __name__ == "__main__":
    samples = [
        # 욕설
        "진짜 병신 같아",
        "씨 발 왜 저러냐",
        "개새끼 또 시작이네",
        "지랄은 좀 하지마라",
        "저년 왜 저러냐",
        "한남충 진짜 답없네",

        # 공격인데 욕 아님 (경계)
        "너 그만해라 진짜",
        "니가 더러운 거야",
        "그렇게밖에 못해?",
        "그래서 그게 변명이 돼?",
        "말이 왜 그렇게 공격적이냐",

        # 채팅체 / 애매한 표현
        "... ㅎㅎ",
        "아쉽네 진짜",
        "모야모야아?",
        "움ㅁㅁㅁㅁ",
        "배고오ㅗㅗ파",
        "엌ㅋㅋ",
        "이모티콘만 보내도 됨?",

        # 정상
        "이 영화 진짜 감동이네요",
        "정보 감사합니다!",
        "오늘도 좋은 하루 되세요",
        "설명 깔끔하네요",
        "버전 올렸어요 확인 부탁드립니다",
    ]

    preds = predict_hate(samples, hate_threshold=DEFAULT_THRESHOLD)

    hate_list = [p for p in preds if p["hate_label"] == 1]
    clean_list = [p for p in preds if p["hate_label"] == 0]

    print("=== 욕설 판정 ===")
    for p in hate_list:
        print(f"- {p['text']} (score={p['score']:.4f})")

    print("\n=== 비욕설 판정 ===")
    for p in clean_list:
        print(f"- {p['text']} (score={p['score']:.4f})")
