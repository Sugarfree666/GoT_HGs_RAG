动机：目前的工作要么是GoT+KGs/RAG，要么就是KHs+multi-hop RAG。可以将GoT和超图知识库结合，Knowledge Hypergraph 是知识层的图，Graph-of-Thoughts 是推理层的图。前者决定“知识如何被组织”，后者决定“推理如何被组织”。具体来说现有方法在“知识表示”上已经走向Hypergraph，但是目前主流的推理范式是CoT思维链或者ToT树状结构，至于PRoH中的DAG，它的多个前驱指向同一个节点是指当前子问题的推理需要前面若干个子问题的答案才能进行。而我们的方法的节点是带状态的推理单元。包含支持的证据，当前的思考，分数，状态等。

普通 RAG：知识结构弱；

KG-RAG：通常只建二元关系；

KH-RAG：知识表示更强，但推理常常还是线性、树式、或有限 DAG；

GoT：推理组织更强，但没有天然绑定 KH 这种高阶知识底座。

> 如何在 KH 上初始化种子 thought
>
> 如何让 thought 图与超图检索交替推进
>
> 如何做多分支证据扩展与汇合
>
> 如何根据中间证据回修当前推理图
>
> 如何让最终答案由多个 reasoning branches 收敛出来



## **数据结构**

ThoughtState定义：

- thought_id:
- role：4种thought类型
- content:当前的thought自然语言表达
- grounding：{anchor，evidence}当前思维节点对应哪些实体超边，chunk。有哪些证据支持
- score：当前thought的评分
- status：当前思维节点的状态
- parent_ids：记录父节点id

ThoughtGraph定义：

- question：原始问题，整个ThoughtGraph是围绕一个问题运行的全局推理状态。
- root_id：根 thought 的 id
- frontier_ids：当前的前沿节点
- status：图的状态，{"running", "done", "failed"}

## **初始框架：**

1. 首先构建超图，参照PRoH的构建方法。

2. 将问题转化成一个结构化任务框架，告诉系统推理方向。作为后续seed思维生成的依据。可以拆成以下结构：

```python
TaskFrame = {
"anchors": ...,
"target": ...,
"constraints": ...,
"bridges": ...,
"hypothesis_template":...
}
```

1. 根据问题q从全局提取一个与问题相关的子图。

   > **如何提取子图？**：根据TaskFrame来指导子图提取，从anchors找出主题实体t；用target和bridges生成目标超边集合r；通过计算余弦相似度匹配超图中的主题实体集 $T$和超边集合 $R$。提取 $T$和 $R$在超图中3-hop领域子图作为问题子图。

2. 根据生成的结构化任务框架初始化一个seed thought，将thought分为以下4种类型：

   1. hypothesis thought：当前局部待验证的猜想

   2. Constraint Thoughts:规定答案必须满足哪些条件

   3. Bridge Thoughts：目标是找到能够连接两段证据的中间桥

   4. Evidence Thought：从 KH 检索到的、可支持前三类 thought 的 grounded evidence

      

3. 根据评分从当前的thought中选择最值得推进的，这一步是对thoughts进行打分，决定保留或者继续推进哪些？

**如何评分？**：
$$
Score(t) = V_{task} \times (1 + V_{grounding})
$$
$V_{task}$：表示当前thought对解决问题有没有帮助。计算公式 $V_{task}=sim(t.content,Q)$

$V_{grounding}$表示在超图中是否有实体和证据支持，计算公式 $V_{grounding}=sim(t.content,t.grounding)$

计算出所有的frontier thought 的基础分，然后用程序粗筛选择top-k个thought，具体来说先看status字段不是activate的，然后筛选得分低于阈值s的。再让LLM来进行精筛选择top-k个thought进行推进。

4. 然后对于选中的thought检索支持或者反驳的证据。检索目标以当前thought可验证可扩展为目标。根据新的证据来进行后续的思维操作。思维操作包含以下4个操作。

   1. Expand：从一个 thought 派生若干新 thought。
   2. Merge：把来自不同分支的 thought 合并成一个新 thought。
   3. Revise：新证据出现后，用来修复原来的thought。
   4. Verify：检查 thought 是否真的成立。

5. 框架的推理层包含控制器，打分模块检索模块，控制器控制着思维节点应该执行哪些操作，打分模块用来对thoughts进行评分排名，检索模块用来在超图上检索证据。控制器的结构应该是“调度器+执行器+图状态更新器”调度器用来说明本轮要执行哪些thought，每个thought执行什么操作；执行器是执行具体操作；图状态更新器是来维护ThoughtGraph。

6. 推理何时终止？

   1. 让TaskFrame成为来帮助系统检验进度的动态"checklist"
   2. 每当图中产生一个高分的 Evidence Thought 尝试将这个证据注册到TaskFrame中对应的未完成的坑位。
   3. 当TaskFrame的所有条件都被满足，控制器执行一个merge操作，目标是将TaskFrame对应的所有Evidence Thought聚合一个Answer Thought。
   4. 将生成的 Answer Thought 连同支持它的证据链，交给一个轻量级的 LLM 提示词进行最终审查。