import pandas as pd

path = r"C:/Users/JL/Desktop/Student projects/Cryptonite/Cryptonite/unsw-nb15/versions/1/UNSW_NB15_training-set.csv"
df = pd.read_csv(path)

print('Columns:', df.columns.tolist())
print('Shape:', df.shape)

num = df.select_dtypes(include='number')
cor = num.corr().abs()

pairs = [
    (cor.columns[i], cor.columns[j], cor.iloc[i, j])
    for i in range(len(cor.columns))
    for j in range(i + 1, len(cor.columns))
]

ranked = sorted(pairs, key=lambda x: x[2], reverse=True)
print('\nTop 10 numeric dependencies:')
for a, b, val in ranked[:10]:
    print(f'{a} / {b}: {val:.4f}')
