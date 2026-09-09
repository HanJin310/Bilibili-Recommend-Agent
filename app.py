# ============================================================
# app.py - B站智能推荐Agent Web 界面
# 基于 Streamlit 构建，提供配置管理、记忆管理、对话交互
# ============================================================

import streamlit as st
import time
import json
import os
from agent import AgentState, build_graph
from langchain_openai import ChatOpenAI

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"


def load_config():
    """从 config.json 加载 API 配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_config(config):
    """保存 API 配置到 config.json"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_memory():
    """从 memory.json 加载用户记忆（偏好标签 + 已推荐视频列表）"""
    default = {"preferred_tags": [], "recommended_videos": []}
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "preferred_tags": data.get("preferred_tags", []),
                    "recommended_videos": data.get("recommended_videos", [])
                }
        except:
            return default
    return default


def save_memory(memory):
    """保存用户记忆到 memory.json"""
    to_save = {
        "preferred_tags": memory.get("preferred_tags", []),
        "recommended_videos": memory.get("recommended_videos", [])
    }
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


# ============================================================
# Streamlit 页面配置
# ============================================================
st.set_page_config(page_title="B站智能推荐Agent", page_icon="🎬", layout="wide")

saved_config = load_config()
saved_memory = load_memory()

# ---------- 初始化 session_state ----------
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "video_results": [],
        "user_preference": "",
        "iteration": 0,
        "original_query": "",
        "is_finished": False,
        "llm_config": saved_config,
        "memory": saved_memory,
        "skip_memory": False
    }
if "is_searching" not in st.session_state:
    st.session_state.is_searching = False
if "llm_initialized" not in st.session_state:
    if saved_config.get("api_key") and saved_config.get("base_url") and saved_config.get("model"):
        st.session_state.llm_initialized = True
    else:
        st.session_state.llm_initialized = False


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # ----- API 配置 -----
    st.header("⚙️ API 配置")
    current_config = st.session_state.state.get("llm_config", {})

    api_key = st.text_input(
        "API Key",
        type="password",
        value=current_config.get("api_key", ""),
        placeholder="sk-xxx"
    )
    base_url = st.text_input(
        "Base URL",
        value=current_config.get("base_url", "https://api.deepseek.com/v1"),
        placeholder="https://api.deepseek.com/v1"
    )
    model_name = st.text_input(
        "模型名称",
        value=current_config.get("model", "deepseek-chat"),
        placeholder="deepseek-chat"
    )

    if st.button("💾 保存配置", use_container_width=True):
        if api_key and base_url and model_name:
            config = {"api_key": api_key, "base_url": base_url, "model": model_name}
            st.session_state.state["llm_config"] = config
            save_config(config)
            st.session_state.llm_initialized = True
            st.success("✅ 配置已保存")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("❌ 请填写完整配置")

    if st.session_state.llm_initialized:
        st.info(f"✅ 当前模型: {model_name}")
    else:
        st.warning("⚠️ 请先配置 API")
    st.divider()

    # ----- 记忆管理 -----
    st.subheader("🧠 记忆管理")
    memory = st.session_state.state.get("memory", {})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("偏好标签", len(memory.get("preferred_tags", [])))
    with col2:
        st.metric("已推荐视频", len(memory.get("recommended_videos", [])))

    # 添加偏好标签
    new_tag = st.text_input("➕ 添加偏好标签")
    if st.button("添加标签", use_container_width=True) and new_tag:
        tags = memory.get("preferred_tags", [])
        if new_tag not in tags:
            tags.append(new_tag)
            memory["preferred_tags"] = tags
            st.session_state.state["memory"] = memory
            save_memory(memory)
            st.success(f"✅ 已添加标签: {new_tag}")
            st.rerun()

    # 清空推荐记忆
    if st.button("🗑️ 清空推荐记忆", use_container_width=True):
        memory["recommended_videos"] = []
        st.session_state.state["memory"] = memory
        save_memory(memory)
        st.success("✅ 已清空推荐记忆")
        st.rerun()

    # 清空所有记忆
    if st.button("🗑️ 清空所有记忆", use_container_width=True):
        default_memory = {"preferred_tags": [], "recommended_videos": []}
        st.session_state.state["memory"] = default_memory
        save_memory(default_memory)
        st.success("✅ 所有记忆已清空")
        st.rerun()

    st.divider()

    # ----- 状态信息 -----
    st.subheader("📊 当前状态")
    st.write(f"**推荐轮次**: {st.session_state.state['iteration']}")
    st.write(f"**原始需求**: {st.session_state.state.get('original_query', '未设置')}")
    st.write(f"**消息数**: {len(st.session_state.state['messages'])}")

    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.state = {
            "messages": [],
            "video_results": [],
            "user_preference": "",
            "iteration": 0,
            "original_query": "",
            "is_finished": False,
            "llm_config": st.session_state.state.get("llm_config", {}),
            "memory": st.session_state.state.get("memory", {}),
            "skip_memory": False
        }
        st.rerun()

    if st.button("🗑️ 清除配置", use_container_width=True):
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        st.session_state.state["llm_config"] = {}
        st.session_state.llm_initialized = False
        st.success("✅ 配置已清除")
        st.rerun()


# ============================================================
# 主区域
# ============================================================
st.title("🎬 B站智能推荐Agent")
st.markdown("输入你想看的视频主题，AI会自动搜索、筛选并推荐最匹配的B站视频")

# ----- 检查配置 -----
config = st.session_state.state.get("llm_config", {})
config_ready = bool(config.get("api_key") and config.get("base_url") and config.get("model"))
if not config_ready:
    st.warning("⚠️ 请先在左侧边栏配置 API Key、Base URL 和模型名称")
    st.stop()

# ----- 渲染历史消息 -----
for msg in st.session_state.state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ----- 输入框 -----
user_input = st.chat_input("输入你想看的视频主题...")

if user_input and not st.session_state.is_searching and config_ready:
    skip_memory = False
    if "再看一次" in user_input or "再看" in user_input:
        skip_memory = True
        st.session_state.state["skip_memory"] = True
        st.info("🔄 正在跳过记忆，重新推荐之前看过的视频...")
    else:
        st.session_state.state["skip_memory"] = False

    # 测试 API 连接
    try:
        llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=0.3
        )
        llm.invoke("test")
    except Exception as e:
        st.error(f"❌ API 连接失败：{str(e)[:100]}")
        st.stop()

    st.session_state.is_searching = True

    # 处理用户输入
    state = st.session_state.state
    if state["original_query"] == "":
        state["original_query"] = user_input
        state["messages"].append({"role": "user", "content": user_input})
    else:
        state["messages"].append({"role": "user", "content": user_input})
        if "换一批" in user_input or "换" in user_input:
            state["video_results"] = []
        if skip_memory:
            state["skip_memory"] = True

    with st.chat_message("user"):
        st.write(user_input)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent 正在思考..."):
            start_time = time.time()
            graph = build_graph(llm)
            result = graph.invoke(state)
            elapsed = time.time() - start_time
            st.caption(f"⏱️ 耗时 {elapsed:.1f} 秒")

            st.session_state.state = result
            if "memory" in result:
                save_memory(result["memory"])

            assistant_reply = result["messages"][-1]["content"]
            st.markdown(assistant_reply)

    st.session_state.is_searching = False
    st.rerun()

st.markdown("---")
st.caption("💡 提示：输入视频主题即可搜索推荐。回复「换一批」获取新推荐，回复「再看一次」可跳过记忆重新推荐。")