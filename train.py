# train.py
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
import joblib


# ================================
# 경로 / 설정
# ================================
# __file__ 이 없는 환경(노트북)에서도 돌아가게 처리
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

# 필요하면 여기 경로를 네 실제 csv 경로로 바꿔도 됨
BINARY_CSV = DATA_DIR / r"C:\Users\pw710\OneDrive\Desktop\ml\ko-toxic-multilabel\dataset\hate_speech_binary_dataset2.csv"
MULTI_CSV  = DATA_DIR / r"C:\Users\pw710\OneDrive\Desktop\ml\ko-toxic-multilabel\dataset\hate_speech_data.csv"

RANDOM_STATE = 42

# TF-IDF 설정 (최종 확정)
VEC_KWARGS = dict(
    analyzer="char",
    ngram_range=(1, 3),
    max_features=300_000,
    min_df=2,
)

# LinearSVC 설정 (최종 확정)
SVC_KWARGS = dict(
    C=0.5,
    class_weight="balanced",
)

VEC_PATH = MODEL_DIR / "vec_char_1_3_300k.joblib"
CLF_PATH = MODEL_DIR / "clf_linearsvc_c05_bal.joblib"


# ================================
# 데이터 로드 및 전처리
# ================================
def load_and_prepare_dataframe() -> pd.DataFrame:
    """
    두 csv를 읽어서 최종 df를 만든다.
    최종 label 정의:
      - 0 = 비욕설/정상
      - 1 = 욕설/혐오
    """
    print("=== [1] CSV 로드 ===")
    print(f"binary: {BINARY_CSV}")
    print(f"multi : {MULTI_CSV}")

    bin_df = pd.read_csv(BINARY_CSV)
    multi_df = pd.read_csv(MULTI_CSV)

    print("binary 컬럼:", list(bin_df.columns))
    print("multi  컬럼:", list(multi_df.columns))

    # 1) binary: 0=혐오/부정, 1=정상/긍정  -> 0=비욕설, 1=욕설로 뒤집기
    if "혐오 여부" not in bin_df.columns or "문장" not in bin_df.columns:
        raise ValueError("binary csv에 '문장' 또는 '혐오 여부' 컬럼이 없습니다.")

    bin_df["label"] = 1 - bin_df["혐오 여부"]

    # 2) multi: 0=정상, 1=혐오 -> 그대로 사용 (이미 1=혐오)
    if "혐오 여부" not in multi_df.columns or "문장" not in multi_df.columns:
        raise ValueError("multi csv에 '문장' 또는 '혐오 여부' 컬럼이 없습니다.")

    multi_df["label"] = multi_df["혐오 여부"]

    # 3) 필요없는 컬럼 제거
    drop_cols = [c for c in multi_df.columns if "Unnamed" in c]
    if drop_cols:
        multi_df = multi_df.drop(columns=drop_cols)

    # 4) 합치기
    df = pd.concat(
        [
            bin_df[["문장", "label"]],
            multi_df[["문장", "label"]],
        ],
        ignore_index=True,
    )

    df["문장"] = df["문장"].astype(str)
    df["label"] = df["label"].astype(int)

    print("\n=== [2] 최종 df 정보 ===")
    print("shape:", df.shape)
    print("label 분포 (0=비욕설, 1=욕설):")
    print(df["label"].value_counts())
    print()

    return df


# ================================
# train / valid / test 분할
# ================================
def split_train_valid_test(df: pd.DataFrame):
    """
    df -> train(80%), valid(10%), test(10%), stratify
    """
    X_all = df["문장"].values
    y_all = df["label"].values

    # 1) train 80%, temp 20%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    # 2) temp 20%를 valid/test 10%/10%로 나누기
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("=== [3] 데이터 분할 결과 ===")
    print(f"train: {len(X_train)}")
    print(f"valid: {len(X_valid)}")
    print(f"test : {len(X_test)}")

    for name, y_split in [("train", y_train), ("valid", y_valid), ("test", y_test)]:
        vc = pd.Series(y_split).value_counts().sort_index()
        ratio0 = vc.get(0, 0) / len(y_split)
        ratio1 = vc.get(1, 0) / len(y_split)
        print(f"[{name}] 0={vc.get(0,0)} ({ratio0:.3f}), 1={vc.get(1,0)} ({ratio1:.3f})")

    print()
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ================================
# 학습 / 평가 / 저장
# ================================
def train_and_evaluate(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
):
    """
    - train으로 학습
    - valid로 성능 확인
    - train+valid로 재학습 후 test 평가
    - 최종 모델/벡터를 저장
    """
    print("=== [4] TF-IDF (char 1~3, 300k) 학습 ===")
    vec = TfidfVectorizer(**VEC_KWARGS)
    X_train_vec = vec.fit_transform(X_train)
    X_valid_vec = vec.transform(X_valid)
    X_test_vec = vec.transform(X_test)

    print("X_train_vec.shape:", X_train_vec.shape)
    print("X_valid_vec.shape:", X_valid_vec.shape)
    print("X_test_vec.shape :", X_test_vec.shape)
    print()

    print("=== [5] LinearSVC (C=0.5, balanced) 학습 ===")
    clf = LinearSVC(**SVC_KWARGS)
    clf.fit(X_train_vec, y_train)

    # valid 성능
    y_valid_pred = clf.predict(X_valid_vec)
    f1_valid = f1_score(y_valid, y_valid_pred)
    print("\n=== [valid] 성능 (threshold=0.0) ===")
    print(f"F1(valid): {f1_valid:.4f}")
    print(classification_report(y_valid, y_valid_pred, target_names=["비욕설", "욕설"]))

    # train+valid로 재학습 후 test 평가
    print("\n=== [6] train+valid 전체로 재학습 후 test 평가 ===")
    X_train_full = np.concatenate([X_train, X_valid], axis=0)
    y_train_full = np.concatenate([y_train, y_valid], axis=0)

    vec_full = TfidfVectorizer(**VEC_KWARGS)
    X_train_full_vec = vec_full.fit_transform(X_train_full)
    X_test_full_vec = vec_full.transform(X_test)

    clf_full = LinearSVC(**SVC_KWARGS)
    clf_full.fit(X_train_full_vec, y_train_full)

    y_test_pred = clf_full.predict(X_test_full_vec)
    f1_test = f1_score(y_test, y_test_pred)
    print(f"\nF1(test): {f1_test:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=["비욕설", "욕설"]))

    # 모델 저장
    print("\n=== [7] 모델 저장 ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec_full, VEC_PATH)
    joblib.dump(clf_full, CLF_PATH)
    print(f"TF-IDF 벡터라이저 저장: {VEC_PATH}")
    print(f"LinearSVC 모델 저장   : {CLF_PATH}")
    print("\n학습 완료.")


def main():
    df = load_and_prepare_dataframe()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(df)
    train_and_evaluate(X_train, X_valid, X_test, y_train, y_valid, y_test)


if __name__ == "__main__":
    main()
