每个 APK 使用独立目录，保存 llm_evidence.json、llm_prompt.md 和可选的 llm_response.md。
先运行两个机器学习解释脚本生成该 APK 的 SHAP、节点注意力、边注意力 CSV，再运行 phase3.llm_evidence 或 phase3.aliyun_llm。
