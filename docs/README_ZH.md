# hilAgent：最简单的Human-in-the-Loop Agent

**hilAgent** 是一个应用plan-and-execute和React思想的最简单的hil agent，其使用了强大的 [langchain/langgraph](https://github.com/hwchase17/langchain)。

## 主要目标
这个项目的主要目标是提供一个"麻雀虽小五脏俱全"的agent示例，体现了人与AI协作的核心原则。

## 特性
- **最小化设计**：专注于基本要素，使其更易于理解和修改。
- **计划执行**：开发并执行计划，可选择人工干预。
- **React哲学**：采用人机协同方法适应变化，确保灵活性和响应性。
- **Self-Criticism节点**：一个专门评估最终结果是否满足需求的节点。
- **搜索工具**：使用TavilySearchResults进行在线搜索。
- **定制化工具**：提供自定义工具examples，包括像check_weather工具这样的自定义程序。

## 前提条件
要开始使用hilAgent，请确保您的系统上已安装以下前提条件：

- Python 3.8或更高版本
- pip
- OpenAI API或其他兼容OpenAI API标准的语音模型，如Kimi或Qwen或Deepseek。
- conda用于环境管理（可选，但推荐）
- TavilySearch API用于在线搜索功能（可选，但推荐）

## 安装
设置hilAgent非常简单。按照以下步骤创建新环境并安装hilAgent：

### 可选但推荐
```bash 
# 创建一个使用Python 3.10的新conda环境
conda create -n hilAgent python=3.10
# 激活hilAgent环境
conda activate hilAgent
```

### 安装hilAgent
```bash
# 克隆此仓库
git clone https://github.com/qingqingjia26/hilAgent.git
# 以可编辑模式安装hilAgent
cd hilAgent
pip install -e .
```

# 使用方法
安装完成后，您可以开始使用hil_agent运行查询。以下是执行查询的一个简单示例：

```bash
hil_agent --query="1+1=?" --llm=""  --base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" --model="qwen2.5-72b-instruct" --api_key=your-api-key 
```

## 人机协同辅助指南

在人机协同模式下，只需输入`-h`即可获得帮助。

关键特性：
- 要跳过指定次数的人工询问，请输入`y -n=<number>`。
- 向LLM提供人工响应，请输入`no your response`。
- 要退出人机协同模式，请输入`exit`。

# 为什么选择hilAgent？
- 学习工具：作为教育资源构建，简化人机协同代理的学习曲线。
- 简单性：易于设置和使用，适合初学者和专家。
- 灵活性：轻松集成到现有系统或作为独立工具使用。

# 贡献
我们欢迎对hilAgent的贡献！如果您发现错误，请通过开启问题来报告。

# 许可证
hilAgent是开源的，并在Apache License Version 2.0许可证下可用。请随意根据需要使用、修改和分发。
