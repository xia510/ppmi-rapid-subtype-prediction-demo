# 本地医学文献知识库

这个目录保存医学文献 RAG 的本地资料。它与患者预测模型相互独立：预测模块回答“模型给出什么概率”，RAG 模块回答“本地文献对一般性科研问题提供了哪些依据”。

## 放什么

把你有权使用的开放获取、指南或自有授权 PDF 放入：

```text
knowledge_base/source_documents/
```

不要放患者病历、检查单、姓名、身份证号、联系方式或其他可识别个人的信息。请保留清晰的文件名和 PDF 元数据标题，网页引用会显示它们。

## 怎样建库

在项目根目录运行：

```powershell
python scripts\build_literature_index.py
```

脚本依次完成：

1. 用 PyMuPDF 逐页提取 PDF 文本；
2. 保留文献名、文件名和页码，把长文本切成带重叠的片段；
3. 用 `BAAI/bge-small-zh-v1.5` 将每个片段转成向量；
4. 保存 `index/embeddings.npy` 和 `index/metadata.json`；
5. 问答时用余弦相似度选出最相关的 Top-k 片段。

PDF 或索引内容变化后必须重新运行建库脚本，并重启 FastAPI。默认目录可用 `PPMI_LITERATURE_INDEX` 修改。

## 为什么不提交 Git

`source_documents/*.pdf` 和 `index/` 已被 `.gitignore` 排除。这样可以避免误提交受版权约束的文献、生成文件和大体积向量。仓库只保存代码和使用说明。

## 回答边界

问答时发送给 DeepSeek 的是“一般性问题 + 本次检索的 Top-k 文献片段”，不发送预测模型、训练数据或患者特征。后端会检查 DeepSeek 引用的 `source_id`，只展示真实检索到的来源及页码。

RAG 可能漏检或误读，向量相似度也不代表证据等级。输出仅供科研文献辅助阅读，不构成临床诊断、治疗或个体医疗建议。
