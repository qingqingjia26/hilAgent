English | [**中文**](https://github.com/qingqingjia26/hilAgent/docs/README_ZH.md)

# hilAgent: The Simplest Human-in-the-Loop Agent

**hilAgent** is a minimalistic human-in-the-loop agent designed for plan-and-execute and reactive operations using the powerful [langchain/langgraph](https://github.com/hwchase17/langchain).

## Key Objective
The primary goal of this project is educational: to offer a "small yet complete" example of an agent that embodies the core principles of human-AI collaboration. 

It's perfect for learners who want to grasp the fundamentals of integrating human decision-making and plan-and-execute framework into AI workflows in a concise and efficient manner.

## Features
- **Minimalistic Design**: Focuses on the essentials, making it easier to understand and modify.
- **Plan-and-Execute**: Develops and carries out plans with the option for human intervention.
- **React Philosophy**: Adapts to changes with a human-in-the-loop approach, ensuring flexibility and responsiveness.
- **Self-Criticism node**:  A module dedicated to evaluating and providing feedback on the agent's final outcomes.
- **Search Tool**: Conduct online searches using TavilySearchResults.
- **Customized Tools**: Seamlessly integrate with additional tools, including custom utilities like the check_weather tool


## Pre-requisites
To get started with hilAgent, ensure you have the following prerequisites installed on your system:
- Python 3.8 or higher
- pip
- OpenAI API or other language models, such as Kimi or Qwen or Deepseek, that are compatible with the OpenAI API standards.
- conda for environment management (optional but recommended)
- TavilySearch API for online search functionality(optional but recommended)

## Installation
Setting up hilAgent is a breeze. Follow these steps to create a new environment and install hilAgent:

### optional but recommended
```bash 
# Create a new conda environment with Python 3.10
conda create -n hilAgent python=3.10
# Activate the hilAgent environment
conda activate hilAgent
```
### Install hilAgent
```bash
# clone this repository
git clone git@github.com:qingqingjia26/hilAgent.git
# Install hilAgent in editable mode
cd hilAgent
pip install -e .
```

# Usage
Once installed, you can start using hil_agent to run queries. Here's a simple example of how to execute a query:

```bash
hil_agent --query="1+1=?" --base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" --model="qwen2.5-72b-instruct" --api_key=your-api-key 
```
## Human-in-the-Loop Assistance Guide

To access help within the human-in-the-loop mode, simply type `-h`.

Key Features:
- To bypass the human inquiry a specified number of times, type `y -n=<number>`.
- To provide a human response to the LLM, type `no your response`.
- To exit the human-in-the-loop mode, type `exit`.

# Why Choose hilAgent?
- Learning Tool: Built as an educational resource to simplify the learning curve of human-in-the-loop agents.
- Simplicity: Easy to set up and use, making it accessible for beginners and experts alike.
- Flexibility: Easily integrate with existing systems or use as a standalone tool.
Contributing

We welcome contributions to hilAgent! If you find a bug, please report it by opening an issue.

# License
hilAgent is open-source and available under the Apache License Version 2.0 License. Feel free to use, modify, and distribute it as needed.