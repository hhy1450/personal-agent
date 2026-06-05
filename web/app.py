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
from src.engine.graph import build_workflow_graph
from src.engine.state import WorkflowState

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

    # Progress placeholders
    status_area = st.empty()
    plan_area = st.empty()
    result_area = st.empty()

    try:
        # Build graph and stream execution node by node
        provider = DeepSeekProvider()
        graph = build_workflow_graph(provider)

        initial_state: WorkflowState = {
            "task": new_task,
            "plan": [],
            "current_step": 0,
            "results": {},
            "final_output": "",
            "errors": [],
            "next_action": "continue",
        }

        # Stream each node's output
        step_num = 0
        final_state = None

        for chunk in graph.stream(initial_state, stream_mode="updates"):
            step_num += 1
            node_name = list(chunk.keys())[0]
            node_data = chunk[node_name]

            # Show current step
            plan = node_data.get("plan", [])
            results = node_data.get("results", {})
            errors = node_data.get("errors", [])

            # Build status message
            node_labels = {
                "planner": "🎯 正在规划任务...",
                "router_conditional": "🔀 正在路由...",
                "researcher": "🔍 Researcher 正在搜索信息...",
                "writer": "✍️ Writer 正在撰写内容...",
                "reviewer_node": "🔎 Reviewer 正在审核结果...",
                "aggregator": "📦 正在汇总结果...",
            }
            label = node_labels.get(node_name, f"⚙️ 执行中: {node_name}")
            status_area.info(f"**Step {step_num}**: {label}")

            # Show plan once available
            if plan:
                lines = []
                for i, s in enumerate(plan):
                    done = str(i) in results
                    icon = "✅" if done else "⏳"
                    lines.append(f"{icon} **{i+1}**. {s.get('description', '')}")
                plan_area.markdown("### 📝 执行计划\n" + "\n".join(lines))

            # Show partial results
            if results:
                parts = ["### 📄 当前结果"]
                for k, v in sorted(results.items()):
                    parts.append(f"**Step {int(k)+1}**:\n{v[:800]}")
                result_area.markdown("\n\n".join(parts))

            # Show errors
            if errors:
                for e in errors[-2:]:  # Show last 2 errors
                    st.warning(f"⚠️ {e.get('detail', str(e))}")

            final_state = node_data

        # Show completion
        if final_state:
            final = final_state.get("final_output", "无输出")
            status_area.empty()
            plan_area.empty()
            result_area.empty()

            # Re-show final plan
            plan = final_state.get("plan", [])
            results = final_state.get("results", {})
            if plan:
                st.subheader("📝 执行计划")
                for i, s in enumerate(plan):
                    done = str(i) in results
                    icon = "✅" if done else "⚠️"
                    st.write(f"{icon} **{i+1}**: {s.get('description', '')}")

            st.divider()
            st.subheader("📄 最终结果")
            st.markdown(final)

            error_list = final_state.get("errors", [])
            if error_list:
                st.warning(f"遇到 {len(error_list)} 个警告")
                for e in error_list:
                    st.error(e.get("detail", str(e)))

        update_task_status(db_task.id, TaskStatus.COMPLETED)
        update_workflow_run(run_record.id, str(final_state) if final_state else "{}", TaskStatus.COMPLETED)
        st.success(f"✅ 任务完成！ID: {db_task.id}")

    except Exception as e:
        status_area.empty()
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
