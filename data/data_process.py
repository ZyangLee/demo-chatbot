import pandas as pd

# 读取 Excel 文件
file_path = '/home/liziang/tx/data/明清文物数据库标签.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)

# 合并所有页面的数据
df = pd.concat(sheets.values(), ignore_index=True)

# 获取名称列
names = df['名称']

# 将名称列存成 txt 文件
output_path = '/home/liziang/tx/data/names.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    for name in names:
        f.write(f"{name}\n")