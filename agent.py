# ============================================================
# agent.py - B站智能推荐 Agent 核心逻辑
#
# 功能概述：
# 1. 接收用户的自然语言需求
# 2. 调用 B站 API 搜索视频
# 3. 通过多维度过滤（播放量、点赞、弹幕评论等）筛选高质量视频
# 4. 使用 LLM 生成个性化的推荐理由
# 5. 自动记忆已推荐的视频，避免重复
# 6. 支持用户添加偏好标签，自动注入搜索
#
# 技术栈：LangGraph（工作流编排）+ LangChain（LLM调用）+ httpx（HTTP请求）
# ============================================================

import httpx
import time
import re
from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_fixed


# ============================================================
# 1. 状态定义
# ============================================================
class AgentState(TypedDict):
    messages: List[Dict[str, str]]          # 对话历史
    video_results: List[Dict[str, Any]]     # 当前推荐的视频列表
    user_preference: str                    # 用户偏好（预留）
    iteration: int                          # 推荐轮次
    original_query: str                     # 用户第一次提问的原始内容
    is_finished: bool                       # 会话是否结束
    llm_config: Dict[str, str]              # API 配置
    memory: Dict[str, Any]                  # 用户记忆 {preferred_tags, recommended_videos}
    skip_memory: bool                       # 是否跳过记忆过滤（"再看一次"时启用）


# ============================================================
# 2. B站 API 辅助函数
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def fetch_bilibili_search(keyword: str, page: int = 1) -> List[Dict]:
    """调用 B站 搜索 API，获取视频列表（带自动重试）"""
    url = "https://api.bilibili.com/x/web-interface/search/type"
    params = {"search_type": "video", "keyword": keyword, "page": page}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, params=params, headers=headers)
        data = resp.json()
        if data["code"] != 0:
            raise Exception(f"B站API错误: {data['code']}")
        return data["data"]["result"]


def fetch_video_cid(bvid: str):
    """通过 bvid 获取视频的 cid（弹幕池ID）"""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            data = resp.json()
            if data["code"] == 0:
                return data["data"].get("cid")
            return None
    except Exception:
        return None


def fetch_video_feedback(bvid: str, max_danmaku: int = 20, max_comments: int = 5) -> Dict[str, List[str]]:
    """
    抓取视频的弹幕和热评样本
    用于 LLM 判断视频是否标题党
    """
    result = {"danmakus": [], "comments": []}
    cid = fetch_video_cid(bvid)
    if not cid:
        return result

    # 抓取弹幕
    danmaku_url = f"https://comment.bilibili.com/{cid}.xml"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(danmaku_url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                danmakus = re.findall(r'<d.*?>(.*?)</d>', resp.text)
                result["danmakus"] = danmakus[:max_danmaku]
    except Exception:
        pass

    # 抓取热评
    reply_url = f"https://api.bilibili.com/x/v2/reply?type=1&oid={cid}&pn=1&nohot=0"
    try:
        time.sleep(0.3)
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(reply_url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com"})
            data = resp.json()
            if data["code"] == 0:
                replies = data.get("data", {}).get("replies", [])
                for r in replies[:max_comments]:
                    content = r.get("content", {}).get("message", "")
                    if content:
                        result["comments"].append(content)
    except Exception:
        pass

    return result


def verify_video_by_feedback(title: str, user_query: str, feedback: Dict[str, List[str]], llm: ChatOpenAI) -> tuple:
    """
    使用 LLM 根据弹幕和评论判断视频内容是否真的符合用户需求
    返回: (是否通过, 判断理由)
    """
    danmakus = feedback.get("danmakus", [])
    comments = feedback.get("comments", [])
    if not danmakus and not comments:
        return True, "无用户反馈数据，默认通过"

    prompt = f"""
你是一个B站视频内容验证专家。用户想看关于「{user_query}」的视频。
标题：「{title}」
弹幕样本：{chr(10).join(['- ' + d for d in danmakus[:15]])}
热评样本：{chr(10).join(['- ' + c for c in comments[:5]])}
请判断：这个视频的内容是否真的在讲「{user_query}」？
只回答：是/否
"""
    try:
        response = llm.invoke(prompt)
        if "是" in response.content:
            return True, response.content
        return False, response.content
    except Exception:
        return True, "LLM验证异常，默认通过"


# ============================================================
# 3. LangGraph 节点函数
# ============================================================

def search_videos(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    搜索节点 - 核心逻辑：
    1. 注入偏好标签
    2. LLM 生成 3 个搜索关键词
    3. 多词多页搜索（3词 × 2页 × 5个 = 最多30个候选）
    4. 去重
    5. 记忆过滤（已推荐的视频）
    6. 质量分排序（播放量50% + 点赞30% + 投币20%）
    7. 播放量粗筛
    8. 深度验证（前3个，弹幕+评论判断标题党）
    9. 轻量验证（第4-8个，标题关键词匹配）
    10. 返回 Top 10
    """
    user_query = state.get("original_query", state["messages"][-1]["content"])
    iteration = state.get("iteration", 0)
    memory = state.get("memory", {})
    skip_memory = state.get("skip_memory", False)

    # ----- 1. 注入偏好标签 -----
    preferred_tags = memory.get("preferred_tags", [])
    if preferred_tags and not skip_memory:
        user_query_with_memory = user_query + " " + " ".join(preferred_tags[:2])
        print(f"🧠 注入偏好标签: {preferred_tags[:2]}")
    else:
        user_query_with_memory = user_query

    # ----- 2. LLM 生成 3 个搜索关键词 -----
    prompt_keywords = f"用户想找：{user_query_with_memory}。请生成3个不同的B站搜索关键词，用逗号分隔，只返回关键词。"
    try:
        keyword_response = llm.invoke(prompt_keywords)
        keywords = [kw.strip() for kw in keyword_response.content.split(",") if kw.strip()]
    except Exception:
        keywords = []
    if len(keywords) < 2:
        keywords = [user_query_with_memory, user_query_with_memory + " 科普", user_query_with_memory + " 热门"]
    print(f"🔍 策略拆解为: {keywords}")

    # ----- 3. 多词多页搜索 -----
    all_videos = []
    for kw in keywords:
        for page in [1, 2]:
            try:
                results = fetch_bilibili_search(kw, page)
                for item in results[:5]:
                    all_videos.append({
                        "title": item["title"].replace("<em class=\"keyword\">", "").replace("</em>", ""),
                        "bvid": item["bvid"],
                        "author": item["author"],
                        "play": item.get("play", 0),
                        "like": item.get("like", 0),
                        "coin": item.get("coin", 0),
                        "url": f"https://www.bilibili.com/video/{item['bvid']}",
                    })
                time.sleep(0.2)
            except Exception as e:
                print(f"⚠️ 搜索失败: {e}")
                continue

    # ----- 4. 去重 -----
    seen = set()
    unique_videos = []
    for v in all_videos:
        if v["bvid"] not in seen:
            seen.add(v["bvid"])
            unique_videos.append(v)
    print(f"📦 去重后 {len(unique_videos)} 个")

    # ----- 5. 记忆过滤（已推荐的视频）-----
    recommended_videos = memory.get("recommended_videos", [])

    filtered_by_memory = []
    for v in unique_videos:
        if v["bvid"] in recommended_videos and not skip_memory:
            print(f"⏭️ 已推荐过: {v['title'][:20]}...")
            continue
        filtered_by_memory.append(v)

    if skip_memory:
        print("🔄 跳过记忆过滤（用户要求再看一次）")
    print(f"🧠 记忆过滤后 {len(filtered_by_memory)} 个")

    # 如果过滤后太少（<3个），放宽条件：从原始列表重新取（不跳过已推荐）
    if len(filtered_by_memory) < 3:
        filtered_by_memory = []
        for v in unique_videos:
            filtered_by_memory.append(v)
        print(f"🔄 放宽记忆过滤，剩余 {len(filtered_by_memory)} 个")

    # ----- 6. 质量分 -----
    for v in filtered_by_memory:
        v["quality_score"] = v["play"] * 0.5 + v["like"] * 0.2 + v.get("coin", 0) * 0.3

    sorted_by_score = sorted(filtered_by_memory, key=lambda x: x["quality_score"], reverse=True)

    # ----- 7. 播放量粗筛 -----
    filtered = [v for v in sorted_by_score if v["play"] >= 300]
    if len(filtered) < 5:
        filtered = [v for v in sorted_by_score if v["play"] >= 100]
    print(f"📊 粗筛后 {len(filtered)} 个")

    # ----- 8-9. 深度验证 + 轻量验证 -----
    verified_videos = []
    top_candidates = filtered[:8]

    for idx, v in enumerate(top_candidates):
        if idx < 3:
            # 深度验证：弹幕+评论
            print(f"🔍 深度验证 #{idx+1}: {v['title'][:30]}...")
            feedback = fetch_video_feedback(v["bvid"])
            if feedback["danmakus"] or feedback["comments"]:
                passed, _ = verify_video_by_feedback(v["title"], user_query, feedback, llm)
                if passed:
                    verified_videos.append(v)
                    print(f"✅ 通过")
                else:
                    print(f"❌ 未通过")
            else:
                v["quality_score"] *= 0.6
                verified_videos.append(v)
                print(f"⚠️ 无反馈，降权")
            time.sleep(0.3)
        else:
            # 轻量验证：标题关键词匹配
            title_lower = v["title"].lower()
            keywords_in_query = re.findall(r'[\u4e00-\u9fff]{2,}', user_query)[:3]
            match_score = sum(1 for kw in keywords_in_query if kw in title_lower)
            if match_score == 0:
                v["quality_score"] *= 0.5
            else:
                v["quality_score"] *= (1 + match_score * 0.1)
            verified_videos.append(v)

    remaining = filtered[8:]
    verified_videos.extend(remaining)
    verified_videos = sorted(verified_videos, key=lambda x: x["quality_score"], reverse=True)

    # ----- 10. 返回 Top 10 -----
    state["video_results"] = verified_videos[:10]
    state["iteration"] = iteration + 1
    return state


def recommend(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    推荐节点：
    1. LLM 精选 5 个视频，生成带播放量/点赞数的推荐语
    2. 自动记录已推荐的视频 BVID 到记忆
    """
    videos = state["video_results"]
    if not videos:
        state["messages"].append({"role": "assistant", "content": "😅 没搜到相关视频，换个关键词试试？"})
        return state

    user_query = state.get("original_query", state["messages"][-1]["content"])
    top_videos = videos[:10]

    # 构造包含播放量、点赞的候选列表
    prompt = f"用户需求：{user_query}\n候选视频：\n"
    for i, v in enumerate(top_videos):
        play = f"{v['play']/10000:.1f}万" if v['play'] > 10000 else str(v['play'])
        like = f"{v['like']/10000:.1f}万" if v['like'] > 10000 else str(v['like'])
        prompt += f"{i+1}. 标题：{v['title']}\n   UP主：{v['author']}，播放量：{play}，点赞数：{like}，链接：{v['url']}\n"

    prompt += """
请精选5个最推荐的，格式：
【精选推荐】
1. 标题：xxx - UP主：xxx - 播放量：xxx - 点赞数：xxx - 链接：xxx
   理由：xxx 
...
【其他备选】
- 标题：xxx - UP主：xxx - 播放量：xxx - 点赞数：xxx - 链接：xxx
注意：每条推荐的播放量和点赞数必须从上方候选视频中复制，不要自己编造。
"""
    try:
        response = llm.invoke(prompt)
        final = response.content + "\n\n---\n💬 如果想换一批，请回复「换一批」；如果想再看一次之前的，请回复「再看一次」。"
    except Exception:
        final = "⚠️ 生成推荐失败，请重试"

    state["messages"].append({"role": "assistant", "content": final})

    # 记录推荐过的视频 BVID
    memory = state.get("memory", {})
    recommended_videos = memory.get("recommended_videos", [])
    for v in top_videos:
        if v["bvid"] not in recommended_videos:
            recommended_videos.append(v["bvid"])
    memory["recommended_videos"] = recommended_videos
    state["memory"] = memory
    print(f"💾 已记录 {len(top_videos)} 个推荐视频")

    return state


# ============================================================
# 4. LangGraph 图构建
# ============================================================

def build_graph(llm: ChatOpenAI):
    """构建 LangGraph 工作流：search → recommend → END"""
    builder = StateGraph(AgentState)
    builder.add_node("search", lambda state: search_videos(state, llm))
    builder.add_node("recommend", lambda state: recommend(state, llm))
    builder.set_entry_point("search")
    builder.add_edge("search", "recommend")
    builder.add_edge("recommend", END)
    return builder.compile()