"""Personal Agent Streamlit Web 界面."""
import sys
import json
import ast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.config.settings import setup_logging
from src.storage.database import (
    init_db, seed_agent_configs,
    create_task, get_task, list_tasks, update_task_status,
    create_workflow_run, update_workflow_run, get_connection,
)
from src.storage.models import TaskStatus
from src.llm.deepseek import DeepSeekProvider
from src.engine.graph import run_workflow

setup_logging()

st.set_page_config(page_title="个人 Agent", page_icon="🤖", layout="wide")


init_db()
seed_agent_configs()

# ===== 侧边栏 =====
st.sidebar.title("🤖 个人 Agent")
st.sidebar.caption("多 Agent 工作流自动化")

st.sidebar.divider()
st.sidebar.subheader("📋 历史任务")

tasks = list_tasks(limit=50)
emoji_map = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
task_options = {
    t.id: f"{emoji_map.get(t.status.value, '')} #{t.id} {t.title[:30]}"
    for t in tasks
}

selected_id = None
if task_options:
    selected_id = st.sidebar.selectbox(
        "选择一个任务查看详情",
        options=[None] + list(task_options.keys()),
        format_func=lambda x: task_options.get(x, "请选择...") if x else "请选择...",
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
        type_emoji = {"research": "🔍 搜索", "write": "✍️ 写作", "review": "🔎 审核"}
        for i, s in enumerate(plan):
            tag = type_emoji.get(s.get("type", ""), "📌")
            st.markdown(f"<small>**{i+1}**. [{tag}] {s.get('description', '')}</small>", unsafe_allow_html=True)

        # 阶段二：执行工作流
        st.divider()
        st.markdown("### ⚙️ 第二步：执行工作流")
        st.info("Agent 正在分步执行，请稍候...")

        result = run_workflow(provider, task_input, plan=plan)

        results = result.get("results", {})
        exec_errors = result.get("errors", [])

        for i in range(len(plan)):
            if str(i) in results:
                st.markdown(f"<small>✅ 步骤 {i+1} 完成</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"<small>❌ 步骤 {i+1} 失败</small>", unsafe_allow_html=True)

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
        s_emoji = emoji_map.get(task.status.value, "")
        st.subheader(f"📋 任务 #{task.id}  {s_emoji} {task.status.value}")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**标题**：{task.title}")
            st.write(f"**创建时间**：{task.created_at[:19]}")
        with c2:
            st.write(f"**更新时间**：{task.updated_at[:19]}")
        st.text_area("任务描述", task.description, height=80, disabled=True)

        # 查询执行结果
        conn = get_connection()
        runs = conn.execute(
            "SELECT state_json, status, started_at, finished_at "
            "FROM workflow_runs WHERE task_id = %s ORDER BY id DESC LIMIT 1",
            (selected_id,),
        ).fetchall()
        conn.close()

        if runs:
            st.divider()
            st.markdown("### 📄 执行结果")
            for run in runs:
                state_json = run["state_json"]
                try:
                    state = json.loads(state_json) if isinstance(state_json, str) else {}
                except json.JSONDecodeError:
                    try:
                        state = ast.literal_eval(state_json) if state_json and state_json != "{}" else {}
                    except (ValueError, SyntaxError):
                        state = {}

                final_output = state.get("final_output", "")
                if isinstance(final_output, str) and final_output:
                    st.markdown(final_output)
                else:
                    st.info("该任务的最终输出不可用")

                run_time = ""
                if run["started_at"] and run["finished_at"]:
                    from datetime import datetime
                    try:
                        t0 = datetime.fromisoformat(run["started_at"])
                        t1 = datetime.fromisoformat(run["finished_at"])
                        run_time = f" | 耗时: {(t1-t0).total_seconds():.0f}秒"
                    except Exception:
                        pass
                st.caption(f"状态: {run['status']}{run_time}")
        else:
            st.info("暂无执行记录")

elif tasks:
    st.divider()
    st.info("👈 从左侧下拉菜单选择历史任务，查看完整执行结果")
