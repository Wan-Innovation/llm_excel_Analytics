import os
import openai
import pandas as pd
import json
import time
import streamlit as st
from io import BytesIO
import tempfile
import requests
from typing import List, Dict, Any

# 导入分块功能模块
import excel_chunker

# 设置页面配置
st.set_page_config(
    page_title="Excel数据分析助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 模型配置
OPENAI_MODELS = {
    'gemini-2.0-flash': {
        'api_key': "sk-Pr1eb9lR0QMuaC*****8to",
        'api_base': "https://api.gptoai.top/v1",
        'temperature': 0.7,
        'provider': 'openai'
    },

    'gpt-4': {
        'api_key': "sk-Pr1eb9lR0QMuaCPUNCeKNur*********to",
        'api_base': "https://api.gptoai.top/v1",
        'temperature': 0.7,
        'provider': 'openai'
    },

    'glm-4-flash': {
        'api_key': "d3ed095dcfc7********i4",
        'api_base': "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        'temperature': 0.7,
        'provider': 'zhipu'  # 智谱AI
    }
}

OLLAMA_MODELS = {
    'deepseek-r1:14b': {
        'url': 'http://localhost:11434',
        'temperature': 0.7,
        'provider': 'ollama'
    },
    'qwen2.5:14b': {
        'url': 'http://localhost:11434',
        'temperature': 0.7,
        'provider': 'ollama'
    },
    'qwq:32b': {
        'url': 'http://localhost:11434',
        'temperature': 0.7,
        'provider': 'ollama'
    }
}

# 合并所有模型选项
ALL_MODELS = {**OPENAI_MODELS, **OLLAMA_MODELS}

# 初始化session state
if 'model_name' not in st.session_state:
    st.session_state.model_name = 'gemini-2.0-flash'

def chat_with_openai(model_config: Dict[str, Any], excel_data: str, question: str) -> str:
    """
    使用OpenAI API回答关于Excel数据的问题
    
    参数:
        model_config: 模型配置
        excel_data: Excel数据的文本表示
        question: 用户问题
        
    返回:
        response: 模型回答
    """
    try:
        # 使用OpenAI客户端API
        client = openai.OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['api_base'],
            timeout=60.0  # 设置60秒超时
        )
        
        # 使用直接的OpenAI客户端调用
        completion = client.chat.completions.create(
            model=st.session_state.model_name,  # 使用选定的模型
            temperature=model_config['temperature'],
            messages=[
                {"role": "system", "content": "你是一个数据分析专家，擅长分析Excel数据并提供洞察。请只基于提供的数据进行分析，不要抱怨数据不完整。"},
                {"role": "user", "content": f"以下是Excel数据:\n\n{excel_data}\n\n请用中文回答这个问题: {question}"}
            ]
        )
        
        # 获取回复
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        st.error(f"OpenAI查询出错: {str(e)}")
        return None  # 出错时返回None

def chat_with_ollama(model_config: Dict[str, Any], excel_data: str, question: str) -> str:
    """
    使用Ollama API回答关于Excel数据的问题
    
    参数:
        model_config: 模型配置
        excel_data: Excel数据的文本表示
        question: 用户问题
        
    返回:
        response: 模型回答
    """
    try:
        # 构建API请求
        prompt = f"""你是一个数据分析专家，擅长分析Excel数据并提供洞察。
        
以下是Excel数据:

{excel_data}

请用中文回答这个问题: {question}"""
        
        # 调用Ollama API
        response = requests.post(
            f"{model_config['url']}/api/generate",
            json={
                "model": st.session_state.model_name,
                "prompt": prompt,
                "temperature": model_config['temperature'],
                "stream": False
            },
            timeout=120  # 增加超时时间，本地模型可能需要更长处理时间
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            st.error(f"Ollama API错误: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Ollama查询出错: {str(e)}")
        return None  # 出错时返回None

def chat_with_zhipu(model_config: Dict[str, Any], excel_data: str, question: str) -> str:
    """
    使用智谱AI的API回答关于Excel数据的问题
    
    参数:
        model_config: 模型配置
        excel_data: Excel数据的文本表示
        question: 用户问题
        
    返回:
        response: 模型回答
    """
    try:
        # 构建智谱API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_config['api_key']}"
        }
        
        payload = {
            "model": "glm-4-flash",  # 使用GLM-4-Flash模型
            "temperature": model_config['temperature'],
            "messages": [
                {"role": "system", "content": "你是一个数据分析专家，擅长分析Excel数据并提供洞察。请只基于提供的数据进行分析，不要抱怨数据不完整。"},
                {"role": "user", "content": f"以下是Excel数据:\n\n{excel_data}\n\n请用中文回答这个问题: {question}"}
            ]
        }
        
        # 调用智谱API
        response = requests.post(
            model_config['api_base'],
            headers=headers,
            json=payload,
            timeout=60  # 60秒超时
        )
        
        if response.status_code == 200:
            result = response.json()
            # 根据智谱API的响应格式提取回答
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            st.error(f"智谱AI API错误: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"智谱AI查询出错: {str(e)}")
        return None  # 出错时返回None

def chat_with_model(excel_data: str, question: str) -> str:
    """
    根据选择的模型使用相应API回答问题
    
    参数:
        excel_data: Excel数据的文本表示
        question: 用户问题
        
    返回:
        response: 模型回答
    """
    model_config = ALL_MODELS.get(st.session_state.model_name)
    
    if not model_config:
        st.error(f"未找到模型配置: {st.session_state.model_name}")
        return None
    
    if model_config['provider'] == 'openai':
        return chat_with_openai(model_config, excel_data, question)
    elif model_config['provider'] == 'ollama':
        return chat_with_ollama(model_config, excel_data, question)
    elif model_config['provider'] == 'zhipu':
        return chat_with_zhipu(model_config, excel_data, question)
    else:
        st.error(f"不支持的提供商: {model_config['provider']}")
        return None

def chunked_excel_analysis(df, question, chunk_by="auto", max_tokens=3000):
    """
    将DataFrame分块，分别分析，然后汇总结果
    
    参数:
        df: 要分析的DataFrame
        question: 用户的问题
        chunk_by: 分块的依据字段
        max_tokens: 每个块的最大token数
    
    返回:
        汇总的分析结果
    """
    # 创建一个进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. 分块
    status_text.text(f"正在按照 '{chunk_by}' 字段对数据进行分块...")
    
    # 使用excel_chunker模块获取分块
    chunks, stats, total_tokens = excel_chunker.get_chunked_dataframes(df, chunk_by, max_tokens)
    
    # 显示分块统计信息
    chunk_stats_expander = st.expander("分块统计信息", expanded=False)
    with chunk_stats_expander:
        st.write(f"总共创建了 {len(chunks)} 个数据块")
        for i, chunk_stat in enumerate(stats["chunks"]):
            if "group" in chunk_stat:
                st.write(f"块 {i+1}: {chunk_stat['group']}, {chunk_stat['rows']} 行, {chunk_stat['tokens']} tokens")
            else:
                st.write(f"块 {i+1}: {chunk_stat['range']}, {chunk_stat['rows']} 行, {chunk_stat['tokens']} tokens")
    
    # 2. 处理每个块
    chunk_results = []
    result_expanders = []
    
    # 为每个块创建一个可折叠区域
    for i in range(len(chunks)):
        result_expanders.append(st.expander(f"块 {i+1} 分析结果", expanded=False))
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"正在分析块 {i+1}/{len(chunks)}...")
        progress_value = (i / len(chunks)) * 0.8  # 留出20%给汇总步骤
        progress_bar.progress(progress_value)
        
        # 转换为文本
        chunk_text = chunk.to_string(index=False)
        
        # 为每个块创建更具体的指示
        chunk_info = stats["chunks"][i].get("group", stats["chunks"][i].get("range", f"部分 {i+1}"))
        
        # 获取分块提示词
        specific_prompt = excel_chunker.get_chunk_prompt(question, chunk_text, chunk_info)
        
        # 分析块
        try:
            result = chat_with_model(chunk_text, specific_prompt)
            
            # 检查结果是否为None
            if result is None:
                error_message = "API请求失败或返回为空，可能由于网络超时或服务不可用"
                with result_expanders[i]:
                    st.error(f"分析块 {i+1} 出错: {error_message}")
                chunk_results.append({
                    "chunk_info": chunk_info,
                    "result": f"分析出错: {error_message}"
                })
            else:
                chunk_results.append({
                    "chunk_info": chunk_info,
                    "result": result
                })
                with result_expanders[i]:
                    st.success(f"块 {i+1} ({chunk_info}) 分析完成")
                    st.markdown(result)
            
            # 添加适当的延迟以避免API速率限制
            if i < len(chunks) - 1:
                time.sleep(1)
        except Exception as e:
            error_message = str(e) if str(e) else "未知错误"
            with result_expanders[i]:
                st.error(f"分析块 {i+1} 出错: {error_message}")
            chunk_results.append({
                "chunk_info": chunk_info,
                "result": f"分析出错: {error_message}"
            })
    
    # 如果所有块都分析失败，返回错误消息
    if all("分析出错" in r["result"] for r in chunk_results):
        status_text.text("分析完成")
        progress_bar.progress(1.0)
        return "所有数据块分析均失败，请检查API连接或稍后重试。"
    
    # 如果只有一个块且分析成功，直接返回结果，不需要汇总
    if len(chunks) == 1 and "分析出错" not in chunk_results[0]["result"]:
        status_text.text("分析完成")
        progress_bar.progress(1.0)
        return chunk_results[0]["result"]
    
    # 3. 汇总结果 - 即使有部分块分析失败，也尝试汇总剩余结果
    status_text.text("正在汇总所有块的分析结果...")
    progress_bar.progress(0.9)  # 90%进度
    
    # 获取汇总提示词
    has_valid_results = any("分析出错" not in r["result"] for r in chunk_results)
    if not has_valid_results:
        status_text.text("分析失败")
        progress_bar.progress(1.0)
        return "所有数据块分析均失败，请检查API连接或稍后重试。"
    
    summary_prompt = excel_chunker.get_summary_prompt(question, chunk_results)
    
    # 汇总已分析的块
    model_config = ALL_MODELS.get(st.session_state.model_name)
    
    try:
        # 可视化状态
        status_text.text("正在生成最终汇总结果...")
        
        if model_config['provider'] == 'openai':
            client = openai.OpenAI(
                api_key=model_config['api_key'],
                base_url=model_config['api_base'],
                timeout=60.0  # 增加超时时间，避免汇总时超时
            )
            
            completion = client.chat.completions.create(
                model=st.session_state.model_name,
                temperature=model_config['temperature'],
                messages=[
                    {"role": "system", "content": "你是一个数据分析专家，擅长综合分析来自多个数据集的结果并提供整体见解。"},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            
            summary = completion.choices[0].message.content
        elif model_config['provider'] == 'ollama':
            # Ollama API调用代码
            # ...现有代码...
            summary = chat_with_ollama(model_config, summary_prompt, summary_prompt)
        elif model_config['provider'] == 'zhipu':
            # 构建汇总提示
            summary_system_prompt = "你是一个数据分析专家，擅长综合分析来自多个数据集的结果并提供整体见解。"
            
            # 调用智谱API进行汇总
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {model_config['api_key']}"
            }
            
            payload = {
                "model": "glm-4-flash",
                "temperature": model_config['temperature'],
                "messages": [
                    {"role": "system", "content": summary_system_prompt},
                    {"role": "user", "content": summary_prompt}
                ]
            }
            
            response = requests.post(
                model_config['api_base'],
                headers=headers,
                json=payload,
                timeout=120  # 增加超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise Exception(f"智谱AI API错误: {response.status_code} - {response.text}")
        else:
            raise Exception(f"不支持的提供商: {model_config['provider']}")
        
        if summary:
            status_text.text("分析完成")
            progress_bar.progress(1.0)
            return summary
        else:
            status_text.text("汇总失败 - 显示单独块结果")
            progress_bar.progress(1.0)
            # 如果汇总返回为空，展示有效的块结果
            valid_results = "\n\n".join([f"【块 {i+1} ({r['chunk_info']})】\n{r['result']}" 
                                       for i, r in enumerate(chunk_results) 
                                       if "分析出错" not in r["result"]])
            return "汇总分析失败。以下是各个块的有效分析结果:\n\n" + valid_results
    except Exception as e:
        status_text.text("汇总出错 - 显示单独块结果")
        progress_bar.progress(1.0)
        st.error(f"汇总结果出错: {str(e)}")
        # 如果汇总失败，返回所有有效块结果的简单连接
        valid_results = "\n\n".join([f"【块 {i+1} ({r['chunk_info']})】\n{r['result']}" 
                                   for i, r in enumerate(chunk_results) 
                                   if "分析出错" not in r["result"]])
        if valid_results:
            return "汇总分析失败。以下是各个块的有效分析结果:\n\n" + valid_results
        else:
            return "所有数据块分析均失败，无法提供任何结果。请检查API连接或稍后重试。"

def load_excel_data(uploaded_file, max_rows=100):
    """
    加载上传的Excel文件并转换为文本格式
    
    参数:
        uploaded_file: 上传的Excel文件对象
        max_rows: 要加载的最大行数
        
    返回:
        excel_data: Excel数据的文本表示
        df: pandas DataFrame对象
        token_count: 数据token计数
    """
    try:
        # 创建临时文件存储上传的Excel
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # 读取Excel文件
        df = pd.read_excel(tmp_path)[:max_rows]
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        excel_data = df.to_string(index=False)
        
        # 计算token数
        token_count = excel_chunker.num_tokens_from_string(excel_data)
        
        return excel_data, df, token_count
    except Exception as e:
        st.error(f"加载Excel文件出错: {str(e)}")
        return None, None, 0

def check_ollama_availability():
    """检查Ollama服务是否可用"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def main():
    # 设置侧边栏参数控制
    with st.sidebar:
        st.title("Excel数据分析助手")
        st.markdown("---")
        
        # 文件上传部分 - 放在最前面
        st.header("1. 上传Excel文件")
        uploaded_file = st.file_uploader("选择一个Excel文件", type=["xlsx", "xls"])
        max_rows = st.number_input("最大加载行数", min_value=10, max_value=1000, value=500, step=10)
        
        # 分块设置部分 - 精简为只显示阈值
        st.header("2. 分块设置")
        max_tokens_threshold = st.number_input("分块阈值 (tokens)", 
                                              min_value=500, max_value=100000, value=100000, step=100,
                                              help="超过此数量的token将触发分块分析")
        
        # 在代码中直接定义块大小，不在前端显示
        max_tokens_per_chunk = 100000  # 直接定义为固定值
        
        st.markdown("---")
        
        # 检查Ollama可用性 - 为模型选择做准备
        ollama_available = check_ollama_availability()
        
        # 模型选择部分 - 放在最后面
        st.header("3. 选择分析模型")
        
        # 分组显示模型
        model_options = list(OPENAI_MODELS.keys())
        if ollama_available:
            model_options += list(OLLAMA_MODELS.keys())
        else:
            st.warning("本地Ollama服务不可用，无法使用本地模型")
        
        selected_model = st.selectbox(
            "选择模型",
            options=model_options,
            index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0,
            help="选择用于分析数据的AI模型"
        )
        
        # 更新session state中的模型
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            st.success(f"已切换到模型: {selected_model}")
        
        # 显示当前模型提供商和能力
        current_model = ALL_MODELS.get(st.session_state.model_name, {})
        if current_model:
            provider = current_model['provider'].upper()
            provider_color = "green" if provider == "OLLAMA" else "blue"
            st.markdown(f"**提供商:** :{provider_color}[{provider}]")
            
            # 显示模型能力说明
            if provider == "OLLAMA":
                st.info("本地模型：速度更快，无需联网，隐私保护更好")
            else:
                st.info("云端模型：能力更强，分析更准确，但需要API")
                
        st.markdown("---")
        st.caption("© 2025 Excel数据分析助手")
    
    # 主界面
    st.title("📊 Excel数据分析助手")
    st.markdown("本项目旨在通过创新的技术方案，让大模型分析excel数据变得简单直观。上传Excel文件并提出您的问题，AI将为您分析数据并提供洞察。")
    
    
    # 检查是否有上传的文件
    if uploaded_file is not None:
        # 加载Excel数据
        with st.spinner("正在加载Excel数据..."):
            excel_data, df, token_count = load_excel_data(uploaded_file, max_rows)
        
        if df is None:
            st.error("无法加载Excel文件，请检查文件格式是否正确。")
            return
        
        # 显示数据预览
        st.subheader("数据预览")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"文件大小: {len(excel_data):,} 字符")
            st.info(f"Token数量: {token_count:,} tokens")
        with col2:
            st.info(f"行数: {len(df):,}")
            st.info(f"列数: {len(df.columns):,}")
        with col3:
            st.info(f"使用模型: {st.session_state.model_name}")
            provider = ALL_MODELS.get(st.session_state.model_name, {}).get('provider', '').upper()
            st.info(f"提供商: {provider}")
        
        # 显示数据表格
        st.dataframe(df.head(5), use_container_width=True)
        
        # 列信息和分块字段选择
        st.subheader("列信息")
        
        # 创建一个数据字典，展示列信息
        col_info = []
        for col in df.columns:
            col_info.append({
                "列名": col,
                "数据类型": str(df[col].dtype),
                "非空值数": df[col].count(),
                "唯一值数": df[col].nunique()
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True)
        
        # 分块字段选择
        chunking_needed = token_count > max_tokens_threshold
        
        if chunking_needed:
            st.warning(f"数据Token数({token_count:,})超过分块阈值({max_tokens_threshold:,})，将使用分块分析。")
            
            # 分块字段选择
            col_options = ["auto"] + list(df.columns)
            chunk_by = st.selectbox(
                "选择分块字段 (推荐选择唯一值在2-10之间的列)",
                options=col_options,
                index=0,
                help="选择'auto'将自动分块，或选择一个列按该列的值分块"
            )
            
            # 如果选择了列，显示该列的唯一值数量
            if chunk_by != "auto" and chunk_by in df.columns:
                unique_count = df[chunk_by].nunique()
                st.info(f"'{chunk_by}'列有 {unique_count} 个唯一值")
                
                # 如果唯一值太多，给出警告
                if unique_count > 20:
                    st.warning(f"该列唯一值较多，可能会产生大量小块。建议选择唯一值较少的列。")
                
                # 显示该列的唯一值分布
                value_counts = df[chunk_by].value_counts().reset_index()
                value_counts.columns = [chunk_by, '计数']
                st.bar_chart(value_counts.set_index(chunk_by))
        else:
            st.success(f"数据大小适中 ({token_count:,} < {max_tokens_threshold:,})，无需分块。")
            chunk_by = "auto"
        
        # 用户问题输入区域 - 减小高度并设置默认值
        st.subheader("提问")
        st.markdown("""
        <style>
            .stTextArea textarea {
                height: 40px !important;
                min-height: 40px !important;
                overflow-y: hidden !important;  /* 隐藏垂直滚动条 */
                resize: none !important;  /* 禁止用户调整大小 */
                padding-top: 8px !important;
                padding-bottom: 8px !important;
                line-height: 1.5 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        question = st.text_area("请输入您的数据分析问题:", 
                               value="请分析数据特征", 
                               height=40)  # 减小高度为60
        analyze_button = st.button("分析数据", type="primary")
        
        # 如果用户点击了分析按钮并输入了问题
        if analyze_button and question:
            st.subheader("分析结果")
            
            # 根据数据大小选择处理方式
            with st.spinner(f"AI ({st.session_state.model_name}) 正在分析您的数据..."):
                if chunking_needed:
                    response = chunked_excel_analysis(df, question, chunk_by, max_tokens_per_chunk)
                else:
                    response = chat_with_model(excel_data, question)
                
                if response:
                    st.markdown(response)
                else:
                    st.error("分析过程中出现错误，请稍后再试。")
    else:
        # 如果用户还没有上传文件
        st.info("请在左侧边栏上传Excel文件以开始分析。")
        
        # 显示功能介绍
        st.markdown("""
        ### 功能介绍
        
        本工具可以帮助您快速分析Excel数据并回答问题，无需编程经验。例如，您可以询问：
        
        - 这些数据的总体趋势是什么？
        - 哪个地区的销售额最高？
        - 数据中有哪些异常值？
        - 按季度统计销售业绩，并分析增长率
        - 根据产品类别分析销售额分布
        
        ### 使用方法
        
        1. 在左侧上传您的Excel文件
        2. 调整分块设置（如有需要）
        3. 选择适合的分析模型
        4. 查看数据预览和列信息
        5. 输入您的分析问题或使用默认问题
        6. 点击"分析数据"，等待AI回答
        
        ### 模型选择说明
        
        - **云端模型**：能力更强，分析更准确，但需要API访问
        - **本地模型**：运行在您自己的计算机上，无需联网，保护隐私，速度更快
        
        对于大型Excel文件，系统会自动对数据进行分块分析，并整合结果。
        您可以在左侧调整分块策略，以获得更准确的分析结果。
        """)

if __name__ == "__main__":
    main() 