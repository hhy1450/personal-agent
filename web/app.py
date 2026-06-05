"""Streamlit Web UI for Personal Agent."""
import sys
from pathlib import Path

# Ensure src is importable
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

# Page config
st.set_page_config(
    page_title="Personal Agent",
    page_icon="🤖",
    layout="wide",
)

# Init DB
init_db()
seed_agent_configs()

# --- Sidebar ---
st.sidebar.title("🤖 Personal Agent")
st.sidebar.caption("多 Agent 工作流自动化")

# Task history
st.sidebar.divider()
st.sidebar.subheader("📋 任务历史")

tasks = list_tasks(limit=50)

if not tasks:
    st.sidebar.caption("暂无任务记录")

selected_task_id = st.sidebar.selectbox(
    "选择任务",
    options=[None] + [t.id for t in tasks],
    format_func=lambda x: f"#{x}" if x else "—",
    label_visibility="collapsed",
)

# --- Main ---
st.title("🤖 Personal Agent")

# New task input
st.subheader("新建任务")
col1, col2 = st.columns([5, 1])
with col1:
    new_task = st.text_input(
        "输入任务描述",
        placeholder="例如：帮我调研 DeepSeek V3 的特点并写一份报告",
        label_visibility="collapsed",
    )
with col2:
    run_btn = st.button("🚀 执行", type="primary", use_container_width=True)

if run_btn and new_task.strip():
    # Create DB record
    db_task = create_task(title=new_task[:100], description=new_task)
    update_task_status(db_task.id, TaskStatus.RUNNING)
    run_record = create_workflow_run(db_task.id)

    try:
        # === Phase 1: Planning ===
        with st.status("🎯 正在规划任务...", expanded=True) as status:
            provider = DeepSeekProvider()

            # Run just the planner first to get the plan
            from src.engine.nodes.planner import PlannerNode
            planner = PlannerNode(provider)
            plan_state = planner({"task": new_task})

            plan = plan_state.get("plan", [])
            errors = plan_state.get("errors", [])

            if errors:
                st.error(f"规划失败: {errors[0].get('detail', str(errors[0]))}")
                st.stop()

            st.write(f"任务已拆解为 **{len(plan)}** 步：")
            for i, step in enumerate(plan):
                emoji = {"research": "🔍", "write": "✍️", "review": "🔎"}.get(step.get("type", ""), "📌")
                st.write(f"  {emoji} **{i+1}**. {step.get('description', '')}")

            status.update(label="✅ 规划完成！开始执行...", state="complete")

        # === Phase 2: Execute full workflow ===
        with st.status("⚙️ 正在执行工作流...", expanded=True) as exec_status:
            result = run_workflow(provider, new_task)

            results = result.get("results", {})
            exec_errors = result.get("errors", [])

            for i, step in enumerate(plan):
                step_key = str(i)
                if step_key in results:
                    st.write(f"✅ **Step {i+1}** 完成")
                else:
                    st.write(f"❌ **Step {i+1}** 失败或跳过")

            if exec_errors:
                exec_status.update(label=f"⚠️ 执行完成，有 {len(exec_errors)} 个警告", state="complete")
            else:
                exec_status.update(label="✅ 执行完成！", state="complete")

        # === Show final result ===
        st.divider()

        # Show plan summary
        st.subheader("📝 执行计划")
        for i, step in enumerate(plan):
            done = str(i) in results
            icon = "✅" if done else "❌"
            st.write(f"{icon} **{i+1}**: {step.get('description', '')}")

        # Show final output
        st.divider()
        st.subheader("📄 最终结果")
        final = result.get("final_output", "无输出")
        st.markdown(final)

        # Show errors
        all_errors = result.get("errors", [])
        if all_errors:
            with st.expander(f"⚠️ {len(all_errors)} 个警告/错误"):
                for e in all_errors:
                    st.warning(e.get("detail", str(e)))

        update_task_status(db_task.id, TaskStatus.COMPLETED)
        update_workflow_run(run_record.id, str(result), TaskStatus.COMPLETED)
        st.success(f"✅ 任务完成！ID: {db_task.id}")

    except Exception as e:
        update_task_status(db_task.id, TaskStatus.FAILED)
        update_workflow_run(run_record.id, "{}", TaskStatus.FAILED)
        st.error(f"❌ 执行失败: {str(e)}")

# Show selected task detail
if selected_task_id:
    st.divider()
    task = get_task(selected_task_id)
    if task:
        status_emoji = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
        st.subheader(f"📋 任务 #{task.id}  {status_emoji.get(task.status.value, '')}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**状态**: {task.status.value}")
            st.write(f"**标题**: {task.title}")
        with col_b:
            st.write(f"**创建时间**: {task.created_at[:19]}")
            st.write(f"**更新时间**: {task.updated_at[:19]}")
        st.text_area("描述", task.description, height=100, disabled=True)

elif tasks:
    st.divider()
    st.info("👈 从左侧下拉菜单选择任务查看详情，或在上方输入新任务")
