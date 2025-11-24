# predict.py
"""
학습된 욕설 분류 모델을 이용한 예측 스크립트

- 사용 전제:
    * train.py 를 먼저 돌려서
      ./models/vec_char_1_3_500k.joblib
      ./models/clf_linearsvc_v1.joblib
      가 존재해야 한다.

- 라벨 규칙:
    * 0 = 비욕설
    * 1 = 욕설

- decision_function 점수:
    * score 가 클수록 '욕설(1)' 쪽에 가깝다고 본다.
    * hate_threshold 이상이면 욕설로 판정.
"""

import argparse
import joblib
import pandas as pd
from typing import List, Union


VEC_PATH = "./models/vec_char_1_3_500k.joblib"
CLF_PATH = "./models/clf_linearsvc_v1.joblib"


# 모델 로드
vec = joblib.load(VEC_PATH)
clf = joblib.load(CLF_PATH)


def predict_hate(
    texts: Union[str, List[str]],
    hate_threshold: float = 0.0,
):
    """
    texts: 문자열 또는 문자열 리스트
    hate_threshold:
        - score >= hate_threshold  ->  욕설(1)
        - score <  hate_threshold  ->  비욕설(0)

    LinearSVC에서 decision_function의 score는
    기본적으로 클래스 1 쪽으로 가까울수록 값이 커진다.
    (y가 {0,1}로 학습되었다고 가정)
    """
    if isinstance(texts, str):
        texts = [texts]

    X = vec.transform(texts)
    scores = clf.decision_function(X)  # array-like

    results = []
    for t, s in zip(texts, scores):
        hate_label = 1 if s >= hate_threshold else 0  # 1=욕설, 0=비욕설

        results.append(
            {
                "text": t,
                "score": float(s),
                "hate_label": int(hate_label),
                "hate_str": "욕설" if hate_label == 1 else "비욕설",
            }
        )
    return results


def classify_csv(
    input_path: str,
    output_path: str,
    hate_threshold: float = 0.0,
    text_column: str = "문장",
):
    """
    CSV 파일을 읽어서 해당 컬럼(text_column)에 대해 욕설 여부를 예측하고,
    score / hate_label / hate_str 컬럼을 추가해서 저장한다.
    """
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"{input_path} 에 '{text_column}' 컬럼이 없습니다.")

    texts = df[text_column].astype(str).tolist()
    preds = predict_hate(texts, hate_threshold=hate_threshold)

    df["score"] = [p["score"] for p in preds]
    df["hate_label"] = [p["hate_label"] for p in preds]
    df["hate_str"] = [p["hate_str"] for p in preds]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="한 문장만 예측해보고 싶을 때 문자열 입력",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="일괄 분류할 입력 CSV 경로 (문장 컬럼 필요)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output_pred.csv",
        help="일괄 분류 결과 저장 경로",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="문장",
        help="CSV에서 문장이 들어있는 컬럼명",
    )
    parser.add_argument(
        "--hate_threshold",
        type=float,
        default=0.0,
        help="욕설 판정 임계값 (기본 0.0, 높일수록 '강한 욕'만 잡음)",
    )
    args = parser.parse_args()

    if args.text is not None:
        # 단일 문장 예측
        res = predict_hate(args.text, hate_threshold=args.hate_threshold)[0]
        print("=== 단일 문장 예측 ===")
        print(f"text      : {res['text']}")
        print(f"score     : {res['score']:.4f}")
        print(f"hate_label: {res['hate_label']} ({res['hate_str']})")

    if args.input_csv is not None:
        # CSV 일괄 예측
        classify_csv(
            input_path=args.input_csv,
            output_path=args.output_csv,
            hate_threshold=args.hate_threshold,
            text_column=args.text_column,
        )

    if args.text is None and args.input_csv is None:
        # 아무 인자도 안 준 경우 샘플 테스트
        print("인자가 없어서 샘플 문장들로 테스트합니다.")
        samples = [
            # 욕설
            "진짜 병신 같아",
            "씨 발 왜 저러냐",
            "개새끼 또 시작이네",
            # 정상
            "이 영화 진짜 감동이네요",
            "설명 깔끔하네요",
            "오늘도 좋은 하루 되세요",
        ]
        preds = predict_hate(samples, hate_threshold=args.hate_threshold)
        for p in preds:
            print(
                f"{p['text']} -> score={p['score']:.4f}, "
                f"label={p['hate_label']}({p['hate_str']})"
            )


if __name__ == "__main__":
    main()
