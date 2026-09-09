# 🎬 B站智能推荐Agent

基于 **LangGraph** 和 **Streamlit** 构建的智能视频推荐系统，支持自然语言搜索、偏好记忆、智能过滤和多轮对话。

## ✨ 功能特点
- 🔍 **自然语言搜索**：输入“猫头上被贴胶布”这类描述性需求，Agent 自动拆解搜索词
- 🧠 **智能记忆**：自动记录推荐过的视频，避免重复推荐
- 🏷️ **偏好标签**：支持用户手动添加偏好标签（如“科普”、“搞笑”），自动注入搜索
- 📊 **数据筛选**：结合播放量、点赞数、弹幕评论进行多维度质量过滤
- 🔄 **多轮交互**：支持“换一批”和“再看一次”指令

## 🛠️ 技术栈
- Python 3.10+
- LangChain / LangGraph (Agent 编排)
- Streamlit (Web UI)
- httpx (B站 API 调用)
- Tenacity (重试机制)

- ## 🚀 快速开始

1. 克隆项目
```bash
git clone https://github.com/你的用户名/Bilibili-Recommend-Agent.git
cd Bilibili-Recommend-Agent
```

2.安装依赖
```bash
pip install -r requirements.txt
```
3.运行应用

```bash
streamlit run app.py
```

4.在侧边栏配置你的 API Key（支持 DeepSeek / OpenAI 等兼容接口）
