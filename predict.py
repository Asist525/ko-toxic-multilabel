# predict.py
import joblib

# 모델 로드
vec = joblib.load("./models/vec_char_1_3_500k.joblib")
clf = joblib.load("./models/clf_linearsvc_v1.joblib")

def predict_hate(texts, hate_threshold=-1.):
    """
    hate_threshold 이하의 점수만 욕설로 본다.
    모델은 1=긍정, 0=그 외로 학습돼 있으니까
    여기서는 점수만 보고 '많이 음수'인 것만 욕설로 친다.
    """
    if isinstance(texts, str):
        texts = [texts]

    X = vec.transform(texts)
    scores = clf.decision_function(X)

    results = []
    for t, s in zip(texts, scores):
        hate_label = 1 if s < hate_threshold else 0
        results.append({
            "text": t,
            "score": float(s),
            "hate_label": int(hate_label),
            "hate_str": "욕설" if hate_label == 1 else "비욕설",
        })
    return results


if __name__ == "__main__":
    samples = [
        # 욕설
        "진짜 병신 같아",
        "씨 발 왜 저러냐",
        "개새끼 또 시작이네",
        "지랄은 좀 하지마라",
        "저년 왜 저러냐",

        # 공격인데 욕 아님
        "너 그만해라 진짜",
        "니가 더러운 거야",
        "그렇게밖에 못해?",
        "그래서 그게 변명이 돼?",
        "말이 왜 그렇게 공격적이냐",

        # 채팅체
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

        # 링크/명령
        "https://www.hankyung.com/article/2023101614447",
        "출발 하삼",
        "절연4 하삼",
    ]

    preds = predict_hate(samples, hate_threshold=-1.3)

    hate_list = [p for p in preds if p["hate_label"] == 1]
    clean_list = [p for p in preds if p["hate_label"] == 0]

    print("=== 욕설 판정 ===")
    for p in hate_list:
        print(f"- {p['text']} (score={p['score']:.4f})")

    print("\n=== 비욕설 판정 ===")
    for p in clean_list:
        print(f"- {p['text']} (score={p['score']:.4f})")
