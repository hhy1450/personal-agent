"""Personal Agent Streamlit Web 界面."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.storage.database import (
    init_db, seed_agent_configs,
    create_task, get_task, list_tasks, update_task_status,
    create_workflow_run, update_workflow_run,
)
from src.storage.models import TaskStatus
from src.llm.deepseek import DeepSeekProvider
from src.engine.graph import run_workflow

st.set_page_config(page_title="个人 Agent", page_icon="🤖", layout="wide")

init_db()
seed_agent_configs()

# ===== 侧边栏 =====
st.sidebar.title("🤖 个人 Agent")
st.sidebar.caption("多 Agent 工作流自动化")

st.sidebar.divider()
st.sidebar.subheader("📋 历史任务")

tasks = list_tasks(limit=50)
task_options = {t.id: f"#{t.id} {t.title[:25]}" for t in tasks}

selected_id = None
if task_options:
    selected_id = st.sidebar.selectbox(
        "选择一个任务查看详情",
        options=[""] + list(task_options.keys()),
        format_func=lambda x: task_options.get(x, "—") if x else "—",
        label_visibility="collapsed",
    )
else:
    st.sidebar.caption("暂无历史记录")

# ===== 主区域 =====
st.title("🤖 个人 Agent")

# --- 新建任务 ---
st.subheader("📝 新建任务")
col1, col2 = st.columns([5, 1])
with col1:
    task_input = st.text_input(
        "任务描述",
        placeholder="例如：帮我调研 DeepSeek V3 的特点并写一份报告",
        label_visibility="collapsed",
    )
with col2:
    go_btn = st.button("🚀 开始执行", type="primary", use_container_width=True)

if go_btn and task_input.strip():
    db_task = create_task(title=task_input[:100], description=task_input)
    update_task_status(db_task.id, TaskStatus.RUNNING)
    run_record = create_workflow_run(db_task.id)

    try:
        provider = DeepSeekProvider()

        # 阶段一：拆解任务
        st.divider()
        st.markdown("### 🎯 第一步：拆解任务")
        st.info("正在调用 AI 分析你的任务...")

        from src.engine.nodes.planner import PlannerNode
        planner = PlannerNode(provider)
        plan_state = planner({"task": task_input})
        plan = plan_state.get("plan", [])
        plan_errors = plan_state.get("errors", [])

        if plan_errors:
            st.error(f"任务拆解失败：{plan_errors[0].get('detail', '')}")
            st.stop()

        st.success(f"任务已拆解为 **{len(plan)}** 个子任务：")
        emoji_map = {"research": "🔍 搜索", "write": "✍️ 写作", "review": "🔎 审核"}
        for i, s in enumerate(plan):
            tag = emoji_map.get(s.get("type", ""), "📌")
            st.write(f"  **{i+1}**. [{tag}] {s.get('description', '')}")

        # 阶段二：执行工作流
        st.divider()
        st.markdown("### ⚙️ 第二步：执行工作流")
        st.info("Agent 正在分步执行，请稍候...")

        result = run_workflow(provider, task_input)

        results = result.get("results", {})
        exec_errors = result.get("errors", [])

        # 显示每步状态
        for i in range(len(plan)):
            if str(i) in results:
                st.success(f"✅ 步骤 {i+1} 完成")
            else:
                st.error(f"❌ 步骤 {i+1} 失败")

        # 阶段三：显示结果
        st.divider()
        st.markdown("### 📄 最终结果")
        final = result.get("final_output", "无输出")
        st.markdown(final)

        if exec_errors:
            with st.expander(f"⚠️ {len(exec_errors)} 个警告"):
                for e in exec_errors:
                    st.warning(e.get("detail", str(e)))

        update_task_status(db_task.id, TaskStatus.COMPLETED)
        update_workflow_run(run_record.id, str(result), TaskStatus.COMPLETED)
        st.success(f"✅ 任务完成！编号: {db_task.id}")

    except Exception as e:
        update_task_status(db_task.id, TaskStatus.FAILED)
        update_workflow_run(run_record.id, "{}", TaskStatus.FAILED)
        st.error(f"❌ 执行失败: {str(e)}")

# ===== 查看历史任务详情 =====
if selected_id:
    st.divider()
    task = get_task(selected_id)
    if task:
        s_emoji = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
        st.subheader(f"📋 任务 #{task.id}  {s_emoji.get(task.status.value, '')}")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**状态**：{task.status.value}")
            st.write(f"**标题**：{task.title}")
        with c2:
            st.write(f"**创建时间**：{task.created_at[:19]}")
            st.write(f"**更新时间**：{task.updated_at[:19]}")
        st.text_area("任务描述", task.description, height=100, disabled=True)

elif tasks:
    st.divider()
    st.info("👈 从左侧下拉菜单选择历史任务查看详情")
