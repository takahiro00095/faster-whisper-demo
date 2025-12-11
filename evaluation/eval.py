import jiwer

# 正解ファイル
ground_truth_file = './out/truth.txt'
# 検証用ファイル
hypothesis_file = './out/large-v3-turbo-int8.txt'

with open(ground_truth_file, 'r', encoding='utf-8') as f:
    ground_truth = f.read()

with open(hypothesis_file, 'r', encoding='utf-8') as f:
    hypothesis = f.read()

# 正規化のための変換処理を定義
normalization = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveWhiteSpace(replace_by_space=False), # 日本語ならスペース除去
    jiwer.RemoveMultipleSpaces(),
])

# 先にテキストを正規化してしまう
ground_truth_normalized = normalization(ground_truth)
hypothesis_normalized = normalization(hypothesis)

# 正規化済みのテキスト同士でCERを計算
cer = jiwer.cer(ground_truth_normalized, hypothesis_normalized)

print(f"CER: {cer}")