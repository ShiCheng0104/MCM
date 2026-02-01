"""
处理收视率数据脚本
功能：
1. 剔除special行
2. 将相同星期的节目归并到同一行
3. 只保留星期数和rating/share, Viewers数据
4. 两期节目合并的在文件名处做标识
"""

import pandas as pd
import os
from datetime import datetime
import re


def parse_date(date_str):
    """解析日期字符串，返回datetime对象"""
    try:
        return datetime.strptime(date_str, "%B %d, %Y")
    except:
        return None


def get_week_number(date_obj, season_start):
    """根据日期计算该季的第几周"""
    if date_obj is None or season_start is None:
        return None
    days_diff = (date_obj - season_start).days
    return days_diff // 7 + 1


def extract_week_from_title(title, allow_non_week=False):
    """从标题中提取周数
    
    Args:
        title: 标题字符串
        allow_non_week: 如果为True，当没有Week信息时返回None而不是过滤
    """
    if pd.isna(title):
        return None
    
    title_str = str(title)
    
    # 先尝试匹配Week数字
    # 匹配 "Week 1", "Week 2: Results" 等格式
    match = re.search(r'Week\s+(\d+)', title_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 匹配 "Top 12 Perform (Week 1)" 格式
    match = re.search(r'\(Week\s+(\d+)\)', title_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 只有在没有Week数字的情况下，才排除特殊节目
    if not allow_non_week:
        special_keywords = ['Road to the Finals', 'Season Finale', 'Anniversary Special', 
                           'Dance-Off', 'Special']
        for keyword in special_keywords:
            if keyword.lower() in title_str.lower():
                return None
    
    return None


def detect_column_names(df):
    """检测表格中的列名，处理不同的命名方式"""
    columns = df.columns.tolist()
    season_num = None
    
    # 检测No列
    no_col = None
    for col in ['No.', 'No', 'Show', 'Episode']:
        if col in columns:
            # 如果是Episode列，需要检查是否是数字序号
            if col == 'Episode':
                # 检查第一个值是否是数字
                try:
                    first_val = df[col].iloc[0]
                    # 使用pd.api.types.is_numeric_dtype或直接尝试转换
                    if pd.api.types.is_numeric_dtype(df[col]):
                        no_col = col
                        break
                except:
                    pass
            else:
                no_col = col
                break
    
    # 检测Title列
    title_col = None
    # 如果Episode列被识别为no_col，则Air date可能是title列
    if no_col == 'Episode' and 'Air date' in columns:
        title_col = 'Air date'
    else:
        for col in ['Title', 'Episode']:
            if col in columns and col != no_col:
                title_col = col
                break
    
    # 检测日期列 - 扩展检测Airdate
    date_col = None
    if title_col != 'Air date':  # 只有当Air date不是title列时才作为日期列
        for col in ['Air date', 'Air Date', 'Airdate']:
            if col in columns:
                date_col = col
                break
    
    # 特殊处理：检测是否是列错位的情况（ratings_21-24）
    # 这些文件的特征：Episode列是数字，Air date是标题，Rating/Share(18–49)是日期
    is_misaligned = False
    if no_col == 'Episode' and title_col == 'Air date':
        # 检查第三列是否看起来像日期
        if 'Rating/Share(18–49)' in columns:
            try:
                third_col_value = str(df['Rating/Share(18–49)'].iloc[0])
                # 如果看起来像日期（包含月份名），说明列错位了
                months = ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
                if any(month in third_col_value for month in months):
                    is_misaligned = True
            except:
                pass
    
    # 检测Rating列（更灵活的匹配）
    rating_col = None
    if is_misaligned:
        # 列错位：Rating实际在Viewers(millions)列
        # 但是如果只有4列，说明没有rating，只有viewers
        if 'Viewers(millions)' in columns:
            # 检查是否有第5列（如18–49 rank或Viewership rank）
            has_fifth_column = len(columns) > 4 and any('rank' in str(col).lower() for col in columns[4:])
            if has_fifth_column:
                # 有5列或更多：第4列是rating
                rating_col = 'Viewers(millions)'
            # 如果只有4列：没有rating，第4列是viewers
    else:
        for col in columns:
            col_lower = col.lower()
            if 'rating' in col_lower and ('share' in col_lower or '/' in col):
                rating_col = col
                break
    
    # 检测Viewers列（更灵活的匹配）
    viewers_col = None
    if is_misaligned:
        # 列错位：检查是否有Viewership rank(Weekly)列或18–49 rank(Weekly)列
        if 'Viewership rank(Weekly)' in columns:
            viewers_col = 'Viewership rank(Weekly)'
        elif '18–49 rank(Weekly)' in columns:
            # ratings_24的情况：viewers在18–49 rank(Weekly)列
            viewers_col = '18–49 rank(Weekly)'
        elif 'Viewers(millions)' in columns:
            # ratings_23的情况：只有4列，第4列Viewers(millions)就是viewers数据
            has_fifth_column = len(columns) > 4
            if not has_fifth_column:
                viewers_col = 'Viewers(millions)'
    else:
        for col in columns:
            col_lower = col.lower()
            # 匹配 Viewers(millions), Viewers (millions), Viewers(million) 等
            if 'viewer' in col_lower and 'million' in col_lower and 'rank' not in col_lower:
                viewers_col = col
                break
    
    return {
        'no': no_col,
        'title': title_col,
        'date': date_col,
        'rating': rating_col,
        'viewers': viewers_col,
        'is_misaligned': is_misaligned
    }


def process_ratings_file(input_file, output_dir='processed'):
    """
    处理单个ratings文件
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 提取季数（从文件名）
    season_num = int(re.search(r'ratings_(\d+)', input_file).group(1))
    
    # 特殊处理：ratings_11已经是处理好的格式，跳过处理
    if season_num == 11:
        print(f"跳过 ratings_11.csv - 已经是处理好的格式")
        print()
        return None
    
    # 检测列名
    col_map = detect_column_names(df)
    
    # 调试输出
    # print(f"  Debug - col_map: {col_map}")
    
    # 剔除special行（如果有No列）
    if col_map['no']:
        df = df[df[col_map['no']] != 'Special'].copy()
    
    # 如果有Title列，剔除特殊节目（不包含周数的）
    if col_map['title']:
        # 标记需要保留的行
        df['has_week_info'] = df[col_map['title']].apply(lambda x: extract_week_from_title(x) is not None)
    
    # 判断是否有Title列且包含Week信息
    has_week_in_title = False
    week_count = 0
    if col_map['title']:
        # 检查是否有Week数字信息（Week 1, Week 2等）
        sample_titles = df[col_map['title']].astype(str)
        # 统计有多少行包含Week信息
        for t in sample_titles:
            if extract_week_from_title(t, allow_non_week=False) is not None:
                week_count += 1
        
        # 如果超过一半的行有Week信息，认为应该从标题提取
        has_week_in_title = week_count > len(df) / 2
    
    # 提取周数
    if has_week_in_title:
        # 从标题中提取周数
        df['week'] = df[col_map['title']].apply(lambda x: extract_week_from_title(x, allow_non_week=False))
        
        # 对于没有Week信息的行，优先用No列（但排除列错位情况）
        if df['week'].isna().any() and col_map['no'] and not col_map.get('is_misaligned', False):
            for idx in df[df['week'].isna()].index:
                no_val = df.loc[idx, col_map['no']]
                if pd.notna(no_val):
                    df.loc[idx, 'week'] = pd.to_numeric(no_val, errors='coerce')
        
        # 如果还有没有week的行，且有日期列，才用日期推算
        if col_map['date'] and df['week'].isna().any():
            df['date_parsed'] = df[col_map['date']].apply(parse_date)
            season_start = df['date_parsed'].min()
            
            # 为没有week的行计算week
            for idx in df[df['week'].isna()].index:
                date_val = df.loc[idx, 'date_parsed']
                if pd.notna(date_val):
                    calculated_week = get_week_number(date_val, season_start)
                    df.loc[idx, 'week'] = calculated_week
        
        # 过滤掉仍然没有week的行
        df = df[df['week'].notna()].copy()
    elif col_map['no']:
        # 优先使用No列作为week（每一行就是一周），避免日期导致的周数跳跃
        df['week'] = pd.to_numeric(df[col_map['no']], errors='coerce')
    elif col_map['date']:
        # 从日期计算周数
        df['date_parsed'] = df[col_map['date']].apply(parse_date)
        season_start = df['date_parsed'].min()
        df['week'] = df.apply(lambda row: get_week_number(row['date_parsed'], season_start), axis=1)
    else:
        # 使用行号作为周数（从1开始）
        df.reset_index(drop=True, inplace=True)
        df['week'] = range(1, len(df) + 1)
    
    # 构建结果DataFrame的列
    result_cols = ['week']
    
    # 添加rating列（如果存在）
    if col_map['rating']:
        result_cols.append('rating_share')
    
    # 添加viewers列（如果存在）
    if col_map['viewers']:
        result_cols.append('viewers')
    
    # 准备数据
    df_clean = df[['week']].copy()
    
    if col_map['rating']:
        df_clean['rating_share'] = df[col_map['rating']]
    
    if col_map['viewers']:
        df_clean['viewers'] = df[col_map['viewers']]
    
    # 移除week为空的行
    df_clean = df_clean.dropna(subset=['week'])
    
    # 按周数分组
    grouped = df_clean.groupby('week')
    
    # 检查是否有合并的情况
    has_merge = any(len(group) > 1 for _, group in grouped)
    
    # 合并相同周的数据
    result_rows = []
    for week, group in grouped:
        if pd.isna(week):
            continue
            
        row = {'week': int(week)}
        
        if len(group) == 1:
            # 单个节目
            if col_map['rating'] and 'rating_share' in group.columns:
                row['rating_share'] = group.iloc[0]['rating_share']
            if col_map['viewers'] and 'viewers' in group.columns:
                row['viewers'] = group.iloc[0]['viewers']
        else:
            # 多个节目合并
            if col_map['rating'] and 'rating_share' in group.columns:
                ratings = group['rating_share'].values
                row['rating_share'] = ', '.join([str(r) for r in ratings if pd.notna(r)])
            
            if col_map['viewers'] and 'viewers' in group.columns:
                viewers = group['viewers'].values
                row['viewers'] = ', '.join([str(v) for v in viewers if pd.notna(v)])
        
        result_rows.append(row)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # 如果结果为空，返回None
    if len(result_df) == 0:
        print(f"警告: {os.path.basename(input_file)} 处理后无有效数据")
        print()
        return None
    
    result_df = result_df.sort_values('week')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    if has_merge:
        output_file = os.path.join(output_dir, f'processed_ratings_{season_num}_merged.csv')
    else:
        output_file = os.path.join(output_dir, f'processed_ratings_{season_num}.csv')
    
    # 保存文件
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"处理完成: {os.path.basename(input_file)}")
    print(f"  - 原始行数: {len(df)}")
    print(f"  - 检测到的列: No={col_map['no']}, Title={col_map['title']}, Date={col_map['date']}")
    if col_map.get('is_misaligned', False):
        print(f"  - 列错位已修正")
    print(f"  - 数据列: Rating={col_map['rating'] is not None}, Viewers={col_map['viewers'] is not None}")
    print(f"  - 合并后周数: {len(result_df)}")
    print(f"  - 输出文件: {os.path.basename(output_file)}")
    print(f"  - 是否有合并: {'是' if has_merge else '否'}")
    print()
    
    return output_file


def process_all_ratings(input_dir='.', output_dir='processed'):
    """
    处理所有ratings文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 获取所有ratings文件
    ratings_files = [f for f in os.listdir(input_dir) if f.startswith('ratings_') and f.endswith('.csv')]
    ratings_files.sort(key=lambda x: int(re.search(r'ratings_(\d+)', x).group(1)))
    
    print(f"找到 {len(ratings_files)} 个ratings文件\n")
    
    output_files = []
    for file in ratings_files:
        input_path = os.path.join(input_dir, file)
        try:
            output_file = process_ratings_file(input_path, output_dir)
            output_files.append(output_file)
        except Exception as e:
            import traceback
            print(f"处理 {file} 时出错: {str(e)}")
            print(f"详细信息: {traceback.format_exc()}\n")
    
    print(f"\n全部处理完成！共处理 {len(output_files)} 个文件")
    print(f"输出目录: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理脚本所在目录下的所有ratings文件
    # 输出到 processed 子目录
    process_all_ratings(input_dir=script_dir, output_dir=os.path.join(script_dir, 'processed'))
