# train.py
"""
한국어 욕설(혐오표현) 분류 모델 학습 스크립트

- 입력 데이터:
    - hate_speech_binary_dataset2.csv
        * 컬럼: ["문장", "혐오 여부"]
        * 라벨: 0=혐오, 1=정상
    - hate_speech_data.csv
        * 컬럼: ["문장", "혐오 여부"] (+ "Unnamed: 0" 같은 인덱스 컬럼)
        * 라벨: 0=정상, 1=혐오

- 통일된 라벨 규칙:
    * label = 0  -> 비욕설
    * label = 1  -> 욕설

- 출력:
    * ./models/vec_char_1_3_500k.joblib
    * ./models/clf_linearsvc_v1.joblib
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib


def load_and_prepare_data(bin_path: str, multi_path: str) -> pd.DataFrame:
    """
    두 CSV를 로드하고 라벨을 통일한 단일 DataFrame 리턴.
    최종 컬럼: ["문장", "label"]
    """
    # 1) 이진 데이터 (0=혐오, 1=정상)
    bin_df = pd.read_csv(bin_path)
    if "문장" not in bin_df.columns or "혐오 여부" not in bin_df.columns:
        raise ValueError(f"{bin_path} 파일의 컬럼명을 확인하세요. (문장, 혐오 여부 필요)")

    # 0=혐오, 1=정상  ->  0=비욕설, 1=욕설 로 맞추기
    # => label = 1 - (혐오 여부)
    bin_df["label"] = 1 - bin_df["혐오 여부"]
    bin_df = bin_df[["문장", "label"]]

    # 2) 멀티 데이터 (0=정상, 1=혐오)
    multi_df = pd.read_csv(multi_path)
    if "문장" not in multi_df.columns or "혐오 여부" not in multi_df.columns:
        raise ValueError(f"{multi_path} 파일의 컬럼명을 확인하세요. (문장, 혐오 여부 필요)")

    # 필요없는 인덱스 컬럼 제거
    drop_cols = [c for c in multi_df.columns if "Unnamed" in c]
    if drop_cols:
        multi_df = multi_df.drop(columns=drop_cols)

    # 0=정상, 1=혐오 -> 그대로 사용 (0=비욕설, 1=욕설)
    multi_df["label"] = multi_df["혐오 여부"]
    multi_df = multi_df[["문장", "label"]]

    # 3) 합치기
    df = pd.concat([bin_df, multi_df], ignore_index=True)

    df["문장"] = df["문장"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def train(
    bin_path: str,
    multi_path: str,
    model_dir: str = "./models",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 데이터 로드 및 전처리
    df = load_and_prepare_data(bin_path, multi_path)
    print("=== 데이터 크기 ===")
    print(df["label"].value_counts())
    print()

    X = df["문장"].values
    y = df["label"].values  # 0=비욕설, 1=욕설

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 벡터라이저 설정
    vec = TfidfVectorizer(
        analyzer="char",       # 문자 단위 n-그램
        ngram_range=(1, 3),    # 1~3글자
        max_features=500_000,  # 피처 수 제한
        min_df=2,              # 너무 희귀한 n-그램 제거
    )

    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # 분류기
    clf = LinearSVC()

    print("=== 학습 시작 ===")
    clf.fit(X_train_vec, y_train)
    print("=== 학습 완료 ===")

    # 간단 평가
    y_pred = clf.predict(X_test_vec)
    print("=== 평가 (classification_report) ===")
    print(classification_report(y_test, y_pred, target_names=["비욕설", "욕설"]))

    # 모델 저장
    os.makedirs(model_dir, exist_ok=True)
    vec_path = os.path.join(model_dir, "vec_char_1_3_500k.joblib")
    clf_path = os.path.join(model_dir, "clf_linearsvc_v1.joblib")

    joblib.dump(vec, vec_path)
    joblib.dump(clf, clf_path)

    print("=== 모델 저장 완료 ===")
    print("Vectorizer:", vec_path)
    print("Classifier:", clf_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin_path",
        type=str,
        default=r"C:\Users\pw710\OneDrive\Desktop\ml\ko-toxic-multilabel\dataset\hate_speech_binary_dataset2.csv",
        help="0=혐오, 1=정상 라벨을 가진 CSV 경로",
    )
    parser.add_argument(
        "--multi_path",
        type=str,
        default=r"C:\Users\pw710\OneDrive\Desktop\ml\ko-toxic-multilabel\dataset\hate_speech_data.csv",
        help="0=정상, 1=혐오 라벨을 가진 CSV 경로",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="모델(.joblib) 저장 디렉토리",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="테스트 비율 (기본 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    train(
        bin_path=args.bin_path,
        multi_path=args.multi_path,
        model_dir=args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
