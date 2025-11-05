# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================
# 설정
# =========================
CSV_PATHS = [
    "./dataset/hate_speech_data.csv",
    "./dataset/hate_speech_binary_dataset2.csv",
]


MODELS_DIR = "./models"
LOG_DIR = "./logs"          
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

NUM_ROUNDS = 3
SEEDS_PER_ROUND = [41, 42, 43, 44, 45]
MAX_FEATURES = 500_000

# =========================
# 1. 데이터 읽기
# =========================
dfs = []
for path in CSV_PATHS:
    df_tmp = pd.read_csv(path)
    if "Unnamed: 0" in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=["Unnamed: 0"])
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

df = df[["문장", "혐오 여부"]]
df = df.dropna(subset=["문장", "혐오 여부"]).copy()

df["혐오 여부"] = (
    df["혐오 여부"]
    .astype(str)
    .str.strip()
    .astype(int)
)

df["문자 개수"] = df["문장"].astype(str).str.len()
df = df[df["문자 개수"] <= 500].reset_index(drop=True)

print("[INFO] initial data shape:", df.shape)
print(df["혐오 여부"].value_counts(dropna=False))

df_for_round = df.copy()

overall_best_acc = 0.0
overall_best_vec = None
overall_best_clf = None

# =========================
# 2. 라운드 반복
# =========================
for round_idx in range(NUM_ROUNDS):
    print(f"\n========== ROUND {round_idx+1}/{NUM_ROUNDS} ==========")

    df_for_round = df_for_round.dropna(subset=["문장", "혐오 여부"]).copy()

    round_best_acc = 0.0
    round_best_artifacts = None

    for seed in SEEDS_PER_ROUND:
        train_df, valid_df = train_test_split(
            df_for_round,
            test_size=0.2,
            random_state=seed,
            stratify=df_for_round["혐오 여부"],
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        y_tr = train_df["혐오 여부"]
        y_va = valid_df["혐오 여부"]

        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            max_features=MAX_FEATURES,
        )
        vec.fit(train_df["문장"])

        X_tr = vec.transform(train_df["문장"])
        X_va = vec.transform(valid_df["문장"])

        clf = LinearSVC(class_weight={0: 1.0, 1: 1.2})
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        print(f"[ROUND {round_idx+1}] seed={seed} acc={acc:.4f}")

        if acc > round_best_acc:
            round_best_acc = acc
            round_best_artifacts = (vec, clf, valid_df, y_pred, y_va, seed)

    vec, clf, valid_df, y_pred, y_va, best_seed = round_best_artifacts
    print(f"[ROUND {round_idx+1}] best seed = {best_seed}, acc = {round_best_acc:.4f}")
    print(classification_report(y_va, y_pred, digits=4))

    wrong_mask = (y_pred != y_va)
    wrong_df = valid_df.loc[wrong_mask].copy()
    wrong_df["y_pred"] = y_pred[wrong_mask]
    wrong_df = wrong_df.dropna(subset=["문장", "혐오 여부"]).copy()

    # 1) 전체 틀린거
    wrong_path = os.path.join(LOG_DIR, f"hard_cases_round{round_idx+1}.csv")
    wrong_df.to_csv(wrong_path, index=False, encoding="utf-8-sig")
    print(f"[ROUND {round_idx+1}] saved all wrong cases -> {wrong_path} ({len(wrong_df)} rows)")

    # 2) 원래 1인데 0으로 감지한 것만
    miss_ones = wrong_df[(wrong_df["혐오 여부"] == 1) & (wrong_df["y_pred"] == 0)].copy()
    miss_path = os.path.join(LOG_DIR, f"hard_missed_label1_round{round_idx+1}.csv")
    miss_ones.to_csv(miss_path, index=False, encoding="utf-8-sig")
    print(f"[ROUND {round_idx+1}] saved missed label=1 -> {miss_path} ({len(miss_ones)} rows)")

    # 3) 원래 0인데 1로 감지한 것만 (재라벨링용)
    false_ones = wrong_df[(wrong_df["혐오 여부"] == 0) & (wrong_df["y_pred"] == 1)].copy()
    false_path = os.path.join(LOG_DIR, f"hard_false_label1_round{round_idx+1}.csv")
    false_ones.to_csv(false_path, index=False, encoding="utf-8-sig")
    print(f"[ROUND {round_idx+1}] saved false label=1 -> {false_path} ({len(false_ones)} rows)")

    # 다음 라운드에 붙일 건 "missed 1"만
    if len(miss_ones) > 0:
        add_df = miss_ones[["문장", "혐오 여부"]].copy()
        df_for_round = pd.concat([df_for_round, add_df], ignore_index=True)
        df_for_round = df_for_round.drop_duplicates(subset=["문장", "혐오 여부"]).reset_index(drop=True)
        print(f"[ROUND {round_idx+1}] next round data shape: {df_for_round.shape}")

    # 전체 베스트 갱신
    if round_best_acc > overall_best_acc:
        overall_best_acc = round_best_acc
        overall_best_vec = vec
        overall_best_clf = clf
        print(f"[ROUND {round_idx+1}] <-- overall best so far")

# =========================
# 3. 베스트 모델 저장
# =========================
joblib.dump(overall_best_vec, os.path.join(MODELS_DIR, "vec_char_1_3_500k.joblib"))
joblib.dump(overall_best_clf, os.path.join(MODELS_DIR, "clf_linearsvc_v1.joblib"))
print(f"\n[INFO] overall best acc: {overall_best_acc:.4f}")
print(f"[INFO] saved best models to {MODELS_DIR}")
print(f"[INFO] logs saved to {LOG_DIR}")
