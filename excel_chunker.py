import pandas as pd
import tiktoken
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 创建控制台对象用于美化输出
console = Console()

def num_tokens_from_string(string, model="gpt-3.5-turbo"):
    """计算字符串中的token数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(string))
    except (ImportError, KeyError):
        # 如果没有tiktoken或无法获取特定模型的编码，使用简单估计
        return len(string) // 4  # 粗略估计：每个token约4个字符

def get_chunked_dataframes(df, chunk_by="auto", max_tokens=3000):
    """
    根据指定的分块策略将DataFrame切分为多个小块
    
    参数:
        df: 要分块的DataFrame
        chunk_by: 按什么字段分块，如果是"auto"则按行数平均分块
        max_tokens: 每个块的最大token数
        
    返回:
        chunks: DataFrame块的列表
        stats: 包含分块统计信息的字典
        total_tokens: 数据总token数
    """
    # 转换为字符串计算总token
    full_text = df.to_string(index=False)
    total_tokens = num_tokens_from_string(full_text)
    
    # 初始化结果
    chunks = []
    stats = {
        "total_rows": len(df),
        "total_tokens": total_tokens,
        "chunks": []
    }
    
    # 如果不需要分块，直接返回整个DataFrame
    if total_tokens <= max_tokens:
        chunks = [df]
        stats["chunks"].append({
            "range": "全部数据",
            "rows": len(df),
            "tokens": total_tokens
        })
        return chunks, stats, total_tokens
    
    # 如果需要按特定字段分块
    if chunk_by != "auto" and chunk_by in df.columns:
        # 按字段分组
        grouped = df.groupby(chunk_by)
        grouped_dfs = []
        
        # 先获取所有组
        for group_name, group_df in grouped:
            grouped_dfs.append((group_name, group_df))
        
        # 当前块信息
        current_chunk = pd.DataFrame(columns=df.columns)
        current_chunk_info = []
        current_tokens = 0
        
        # 遍历每个组
        for group_name, group_df in grouped_dfs:
            # 计算当前组的token数
            group_text = group_df.to_string(index=False)
            group_tokens = num_tokens_from_string(group_text)
            
            # 如果当前组本身就超过了最大token数，需要单独作为一个或多个块
            if group_tokens > max_tokens:
                # 如果当前块不为空，先保存当前块
                if len(current_chunk) > 0:
                    chunks.append(current_chunk.copy())
                    chunk_text = current_chunk.to_string(index=False)
                    chunk_tokens = num_tokens_from_string(chunk_text)
                    stats["chunks"].append({
                        "group": f"混合组: {', '.join(current_chunk_info)}",
                        "rows": len(current_chunk),
                        "tokens": chunk_tokens
                    })
                    # 重置当前块
                    current_chunk = pd.DataFrame(columns=df.columns)
                    current_chunk_info = []
                    current_tokens = 0
                
                # 大组需要拆分成多个小块
                remaining_df = group_df.copy()
                sub_chunk_index = 0
                
                while len(remaining_df) > 0:
                    # 估算每块可容纳的行数
                    rows_per_subchunk = max(1, int(len(remaining_df) * (max_tokens / group_tokens)))
                    
                    # 确保至少有一行
                    if rows_per_subchunk == 0:
                        rows_per_subchunk = 1
                        
                    # 防止超出剩余行数
                    if rows_per_subchunk > len(remaining_df):
                        rows_per_subchunk = len(remaining_df)
                    
                    # 创建子块
                    sub_chunk = remaining_df.iloc[:rows_per_subchunk].copy()
                    sub_chunk_text = sub_chunk.to_string(index=False)
                    sub_chunk_tokens = num_tokens_from_string(sub_chunk_text)
                    
                    # 将子块添加到结果中
                    chunks.append(sub_chunk)
                    sub_chunk_index += 1
                    stats["chunks"].append({
                        "group": f"{group_name} (部分 {sub_chunk_index})",
                        "rows": len(sub_chunk),
                        "tokens": sub_chunk_tokens
                    })
                    
                    # 从剩余数据中移除已处理的行
                    remaining_df = remaining_df.iloc[rows_per_subchunk:].copy()
                    
                    # 重新计算剩余数据的token数
                    if len(remaining_df) > 0:
                        remaining_text = remaining_df.to_string(index=False)
                        group_tokens = num_tokens_from_string(remaining_text)
            else:
                # 当前组不超过最大token数，检查是否可以添加到当前块
                combined_df = pd.concat([current_chunk, group_df])
                combined_text = combined_df.to_string(index=False)
                combined_tokens = num_tokens_from_string(combined_text)
                
                # 如果添加后会超过限制，保存当前块，然后开始新块
                if combined_tokens > max_tokens and len(current_chunk) > 0:
                    chunks.append(current_chunk.copy())
                    chunk_text = current_chunk.to_string(index=False)
                    chunk_tokens = num_tokens_from_string(chunk_text)
                    stats["chunks"].append({
                        "group": f"混合组: {', '.join(current_chunk_info)}",
                        "rows": len(current_chunk),
                        "tokens": chunk_tokens
                    })
                    # 重置为当前组
                    current_chunk = group_df.copy()
                    current_chunk_info = [str(group_name)]
                    current_tokens = group_tokens
                else:
                    # 可以合并，添加到当前块
                    current_chunk = combined_df.copy()
                    current_chunk_info.append(str(group_name))
                    current_tokens = combined_tokens
        
        # 添加最后一个未满的块
        if len(current_chunk) > 0:
            chunks.append(current_chunk.copy())
            chunk_text = current_chunk.to_string(index=False)
            chunk_tokens = num_tokens_from_string(chunk_text)
            stats["chunks"].append({
                "group": f"混合组: {', '.join(current_chunk_info)}" if len(current_chunk_info) > 1 else current_chunk_info[0],
                "rows": len(current_chunk),
                "tokens": chunk_tokens
            })
    else:
        # 自动分块 - 根据token数估算需要的块数
        estimated_chunks_count = math.ceil(total_tokens / (max_tokens * 0.9))  # 使用90%的最大token来估算
        
        if estimated_chunks_count <= 1:
            # 如果只需要一个块，直接返回整个DataFrame
            chunks = [df]
            stats["chunks"].append({
                "range": "全部数据",
                "rows": len(df),
                "tokens": total_tokens
            })
        else:
            # 根据估算的块数均匀分配行
            rows_per_chunk = max(1, len(df) // estimated_chunks_count)
            
            # 确保至少有一行
            if rows_per_chunk == 0:
                rows_per_chunk = 1
            
            # 创建多个块
            for i in range(0, len(df), rows_per_chunk):
                end_idx = min(i + rows_per_chunk, len(df))
                chunk = df.iloc[i:end_idx].copy()
                
                # 计算实际token数
                chunk_text = chunk.to_string(index=False)
                chunk_tokens = num_tokens_from_string(chunk_text)
                
                # 如果这个块超过最大token数，需要进一步分割
                if chunk_tokens > max_tokens and end_idx - i > 1:  # 确保至少有2行可分
                    # 递归调用自身，进一步分割这个块
                    sub_chunks, sub_stats, _ = get_chunked_dataframes(chunk, "auto", max_tokens)
                    chunks.extend(sub_chunks)
                    
                    # 更新统计信息，修改范围描述
                    for sub_stat in sub_stats["chunks"]:
                        if "range" in sub_stat:
                            # 调整范围描述，考虑原始数据中的索引偏移
                            range_parts = sub_stat["range"].split(" 到 ")
                            if len(range_parts) > 1 and range_parts[0].startswith("行 "):
                                start_row = int(range_parts[0].replace("行 ", ""))
                                end_row = int(range_parts[1])
                                sub_stat["range"] = f"行 {start_row + i} 到 {end_row + i}"
                        stats["chunks"].append(sub_stat)
                else:
                    # 块大小合适，直接添加
                    chunks.append(chunk)
                    stats["chunks"].append({
                        "range": f"行 {i+1} 到 {end_idx}",
                        "rows": len(chunk),
                        "tokens": chunk_tokens
                    })
    
    return chunks, stats, total_tokens

def print_chunk_stats(chunks, stats):
    """打印分块统计信息"""
    # 创建一个漂亮的表格来显示分块信息
    chunk_table = Table(title="分块统计", show_header=True, header_style="bold cyan")
    chunk_table.add_column("块", style="cyan")
    chunk_table.add_column("范围/组", style="green")
    chunk_table.add_column("行数", style="yellow", justify="right")
    chunk_table.add_column("Token数", style="yellow", justify="right")
    
    for i, chunk_stat in enumerate(stats["chunks"]):
        if "group" in chunk_stat:
            chunk_table.add_row(
                f"{i+1}", 
                chunk_stat["group"], 
                f"{chunk_stat['rows']:,}", 
                f"{chunk_stat['tokens']:,}"
            )
        else:
            chunk_table.add_row(
                f"{i+1}", 
                chunk_stat["range"], 
                f"{chunk_stat['rows']:,}", 
                f"{chunk_stat['tokens']:,}"
            )
    
    console.print(chunk_table)
    console.print(f"[green]总行数: {stats['total_rows']:,}[/green]")
    console.print(f"[green]总Token数: {stats['total_tokens']:,}[/green]")
    console.print(f"[green]块数: {len(chunks)}[/green]")

def get_chunk_prompt(question, chunk_text, chunk_info):
    """为特定的块生成提示词"""
    return f"""这是数据的一个子集（{chunk_info}）：

{chunk_text}

请分析这部分数据回答以下问题，但明确说明你只分析了部分数据：{question}

你的分析应该专注于这个数据子集，但可以提出可能需要查看完整数据才能得出的结论。"""

def get_summary_prompt(question, chunk_results):
    """生成汇总提示词"""
    chunks_analysis = "\n\n".join([
        f"数据块 {i+1} ({r['chunk_info']}) 分析结果:\n{r['result']}"
        for i, r in enumerate(chunk_results)
        if "分析出错" not in r["result"]
    ])
    
    return f"""我已经将一个大型数据集分成了几个块，并分别进行了分析。现在我需要你综合这些分析结果，提供一个全面的回答。

原始问题是: {question}

以下是每个数据块的分析结果:

{chunks_analysis}

请综合所有块的分析，提供一个完整、连贯的回答。识别共同的模式、趋势和见解，解决任何矛盾，并提供对整个数据集的全面理解。你的回答应该直接针对原始问题，不需要重复每个块的具体分析过程。""" 