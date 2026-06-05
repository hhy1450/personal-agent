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

selected_task_id = None
for t in tasks:
    emoji = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}.get(t.status.value, "❓")
    label = f"{emoji} {t.title[:30]}"
    if st.sidebar.button(label, key=f"task_{t.id}", use_container_width=True):
        selected_task_id = t.id

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
    with st.spinner("正在执行任务..."):
        # Create DB record
        db_task = create_task(title=new_task[:100], description=new_task)
        update_task_status(db_task.id, TaskStatus.RUNNING)
        run_record = create_workflow_run(db_task.id)

        try:
            provider = DeepSeekProvider()
            result = run_workflow(provider, new_task)

            # Show plan
            plan = result.get("plan", [])
            results = result.get("results", {})

            st.subheader("📝 执行计划")
            for i, step in enumerate(plan):
                done = str(i) in results
                icon = "✅" if done else "⏳"
                st.write(f"{icon} **Step {i+1}**: {step.get('description', '')}")

            # Show final output
            final = result.get("final_output", "无输出")
            st.divider()
            st.subheader("📄 最终结果")
            st.markdown(final)

            # Show errors
            errors = result.get("errors", [])
            if errors:
                st.warning(f"遇到 {len(errors)} 个警告")
                for e in errors:
                    st.error(e.get("detail", str(e)))

            update_task_status(db_task.id, TaskStatus.COMPLETED)
            update_workflow_run(run_record.id, str(result), TaskStatus.COMPLETED)
            st.success(f"任务完成！ID: {db_task.id}")
            st.rerun()

        except Exception as e:
            update_task_status(db_task.id, TaskStatus.FAILED)
            update_workflow_run(run_record.id, "{}", TaskStatus.FAILED)
            st.error(f"执行失败: {str(e)}")

# Show selected task detail
if selected_task_id:
    st.divider()
    task = get_task(selected_task_id)
    if task:
        st.subheader(f"📋 任务 #{task.id}")
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
    st.info("👈 点击左侧任务列表查看详情，或在上方输入新任务")
