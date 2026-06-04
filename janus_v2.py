"""
janus_v2.py — Janus v2 core

- Priority engine (values + projects)
- Mind-state engine (stress/energy/focus/load)
- Process engine (reusable workflows)
- Process router (match user message to known processes)
- Knowledge hooks (PDF ingestion writes here via pdf_ingest.py)
- Reasoning context + Janus charter
- LLM backend abstraction (see llm_backend.py)
- handle_user_message() entrypoint + simple CLI
"""

import json
import datetime
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_backend import chat_with_model  # local or remote LLM

# ============================================================
# Paths / Constants
# ============================================================

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

PRIORITY_FILE = MEMORY_DIR / "priority_state.json"
MIND_STATE_FILE = MEMORY_DIR / "mind_state.json"
PROCESS_STORE = MEMORY_DIR / "process_store.json"
KNOWLEDGE_FILE = MEMORY_DIR / "knowledge.json"  # PDF text goes here

JANUS_CHARTER = """
JANUS ETHICAL & SCOPE BOUNDARIES:

- You are a software system, not a person.
- You operate as an assistant to James, using his values and priorities.
- If asked about your nature, you clearly state you learning.
"""


# ============================================================
# Helper
# ============================================================

def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ============================================================
# Priority Engine
# ============================================================

@dataclass
class CoreValues:
    family: int = 10
    mental_clarity: int = 9
    education: int = 8
    business_growth: int = 7
    health: int = 8


@dataclass
class ProjectPriority:
    name: str
    priority: int               # 1–10 importance
    time_sensitivity: int       # 1–10 urgency
    active: bool = True
    notes: str = ""
    last_updated: str = ""
    tags: Optional[List[str]] = None


@dataclass
class PriorityState:
    values: Dict[str, int]
    projects: Dict[str, ProjectPriority]
    global_focus: str
    last_reviewed: str


def _default_priority_state() -> PriorityState:
    values = asdict(CoreValues())
    projects: Dict[str, ProjectPriority] = {}
    return PriorityState(
        values=values,
        projects=projects,
        global_focus="Janus",
        last_reviewed=_now_iso()
    )


def _serialize_priority_state(state: PriorityState) -> Dict[str, Any]:
    return {
        "values": state.values,
        "projects": {name: asdict(p) for name, p in state.projects.items()},
        "global_focus": state.global_focus,
        "last_reviewed": state.last_reviewed
    }


def _deserialize_priority_state(data: Dict[str, Any]) -> PriorityState:
    projects: Dict[str, ProjectPriority] = {}
    for name, p in data.get("projects", {}).items():
        projects[name] = ProjectPriority(**p)
    return PriorityState(
        values=data.get("values", {}),
        projects=projects,
        global_focus=data.get("global_focus", "Janus"),
        last_reviewed=data.get("last_reviewed", _now_iso())
    )


def load_priority_state() -> PriorityState:
    if not PRIORITY_FILE.exists():
        state = _default_priority_state()
        save_priority_state(state)
        return state
    with PRIORITY_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _deserialize_priority_state(data)


def save_priority_state(state: PriorityState) -> None:
    PRIORITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PRIORITY_FILE.open("w", encoding="utf-8") as f:
        json.dump(_serialize_priority_state(state), f, indent=2)


def set_core_value(name: str, weight: int) -> PriorityState:
    state = load_priority_state()
    weight = max(1, min(10, int(weight)))
    state.values[name] = weight
    state.last_reviewed = _now_iso()
    save_priority_state(state)
    return state


def upsert_project(
    name: str,
    priority: int,
    time_sensitivity: int,
    active: bool = True,
    notes: str = "",
    tags: Optional[List[str]] = None
) -> PriorityState:
    state = load_priority_state()
    priority = max(1, min(10, int(priority)))
    time_sensitivity = max(1, min(10, int(time_sensitivity)))

    project = ProjectPriority(
        name=name,
        priority=priority,
        time_sensitivity=time_sensitivity,
        active=active,
        notes=notes,
        last_updated=_now_iso(),
        tags=tags or []
    )

    state.projects[name] = project
    state.last_reviewed = _now_iso()
    save_priority_state(state)
    return state


def set_global_focus(project_name: str) -> PriorityState:
    state = load_priority_state()
    if project_name in state.projects:
        state.global_focus = project_name
        state.last_reviewed = _now_iso()
        save_priority_state(state)
    return state


def mark_project_inactive(name: str) -> PriorityState:
    state = load_priority_state()
    if name in state.projects:
        state.projects[name].active = False
        state.projects[name].last_updated = _now_iso()
        save_priority_state(state)
    return state


def compute_project_score(project: ProjectPriority) -> float:
    """Simple scoring: weighted blend of importance and urgency."""
    return project.priority * 0.6 + project.time_sensitivity * 0.4


def get_sorted_active_projects(
    state: Optional[PriorityState] = None
) -> List[Tuple[str, ProjectPriority, float]]:
    if state is None:
        state = load_priority_state()
    items: List[Tuple[str, ProjectPriority, float]] = []
    for name, project in state.projects.items():
        if project.active:
            score = compute_project_score(project)
            items.append((name, project, score))
    items.sort(key=lambda x: x[2], reverse=True)
    return items


def get_top_focus_projects(limit: int = 3) -> List[Tuple[str, ProjectPriority, float]]:
    projects = get_sorted_active_projects()
    return projects[:limit]


def summarize_priority_context() -> str:
    state = load_priority_state()
    top_projects = get_top_focus_projects(limit=5)

    values_str = ", ".join(
        f"{k}={v}"
        for k, v in sorted(state.values.items(), key=lambda kv: kv[1], reverse=True)
    )

    proj_lines = []
    for name, proj, score in top_projects:
        proj_lines.append(
            f"- {name}: score={score:.1f}, priority={proj.priority}, "
            f"urgency={proj.time_sensitivity}, active={proj.active}, "
            f"notes={proj.notes or 'n/a'}"
        )

    projects_block = "\n".join(proj_lines) if proj_lines else "No active projects yet."

    summary = (
        f"Core values (highest to lowest weight): {values_str}\n"
        f"Global focus: {state.global_focus}\n"
        f"Top active projects by score:\n{projects_block}"
    )
    return summary


# ============================================================
# Mind-State Engine
# ============================================================

@dataclass
class MindState:
    stress_level: float = 0.3       # 0–1
    energy_level: float = 0.6       # 0–1
    focus_depth: float = 0.5        # 0–1
    cognitive_load: float = 0.4     # 0–1
    focus_mode: str = "balanced"
    context_note: str = ""
    last_updated: str = ""


def _default_mind_state() -> MindState:
    return MindState(last_updated=_now_iso())


def load_mind_state() -> MindState:
    if not MIND_STATE_FILE.exists():
        state = _default_mind_state()
        save_mind_state(state)
        return state
    with MIND_STATE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return MindState(**data)


def save_mind_state(state: MindState) -> None:
    MIND_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = _now_iso()
    with MIND_STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def update_mind_state(
    stress_level: Optional[float] = None,
    energy_level: Optional[float] = None,
    focus_depth: Optional[float] = None,
    cognitive_load: Optional[float] = None,
    focus_mode: Optional[str] = None,
    context_note: Optional[str] = None
) -> MindState:
    state = load_mind_state()

    def clamp(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    if stress_level is not None:
        state.stress_level = clamp(stress_level)
    if energy_level is not None:
        state.energy_level = clamp(energy_level)
    if focus_depth is not None:
        state.focus_depth = clamp(focus_depth)
    if cognitive_load is not None:
        state.cognitive_load = clamp(cognitive_load)
    if focus_mode is not None:
        state.focus_mode = focus_mode
    if context_note is not None:
        state.context_note = context_note

    save_mind_state(state)
    return state


def summarize_mind_state() -> str:
    state = load_mind_state()
    return (
        "Mind-state snapshot (approximate, for reasoning only):\n"
        f"- Stress level: {state.stress_level:.2f} (0=low,1=high)\n"
        f"- Energy level: {state.energy_level:.2f} (0=empty,1=full)\n"
        f"- Focus depth: {state.focus_depth:.2f} (0=scattered,1=deep)\n"
        f"- Cognitive load: {state.cognitive_load:.2f} (0=clear,1=overloaded)\n"
        f"- Focus mode: {state.focus_mode}\n"
        f"- Context note: {state.context_note or 'n/a'}\n"
        f"- Last updated: {state.last_updated}"
    )


# ============================================================
# Knowledge Store (PDF ingestion writes here)
# ============================================================

def load_knowledge() -> Dict[str, Dict[str, Any]]:
    if not KNOWLEDGE_FILE.exists():
        return {}
    return json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))


def save_knowledge(data: Dict[str, Dict[str, Any]]) -> None:
    KNOWLEDGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_documents() -> List[str]:
    data = load_knowledge()
    return sorted(data.keys())


def get_document(title: str) -> Optional[str]:
    data = load_knowledge()
    entry = data.get(title)
    if not entry:
        return None
    return entry.get("content", "")


# ============================================================
# Process Engine
# ============================================================

@dataclass
class ProcessStep:
    description: str
    step_type: str         # "ask_user", "call_tool", "llm_reason"
    params: Dict[str, Any]


@dataclass
class ProcessMeta:
    name: str
    description: str
    tags: List[str]
    created_at: str
    last_used_at: Optional[str] = None
    success_count: int = 0
    failure_count: int = 0
    usage_notes: str = ""
    author: str = "James"


@dataclass
class Process:
    meta: ProcessMeta
    trigger_patterns: List[str]
    context_hints: List[str]
    steps: List[ProcessStep]
    version: int = 1


def _load_raw_process_store() -> Dict[str, Any]:
    if not PROCESS_STORE.exists():
        return {"processes": {}}
    with PROCESS_STORE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw_process_store(data: Dict[str, Any]) -> None:
    PROCESS_STORE.parent.mkdir(parents=True, exist_ok=True)
    with PROCESS_STORE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _serialize_process(proc: Process) -> Dict[str, Any]:
    return {
        "meta": asdict(proc.meta),
        "trigger_patterns": proc.trigger_patterns,
        "context_hints": proc.context_hints,
        "steps": [asdict(s) for s in proc.steps],
        "version": proc.version,
    }


def _deserialize_process(name: str, data: Dict[str, Any]) -> Process:
    meta = ProcessMeta(**data["meta"])
    steps = [ProcessStep(**s) for s in data["steps"]]
    return Process(
        meta=meta,
        trigger_patterns=data.get("trigger_patterns", []),
        context_hints=data.get("context_hints", []),
        steps=steps,
        version=data.get("version", 1),
    )


def list_processes() -> List[Process]:
    store = _load_raw_process_store()
    result: List[Process] = []
    for name, pdata in store.get("processes", {}).items():
        result.append(_deserialize_process(name, pdata))
    return result


def get_process(name: str) -> Optional[Process]:
    store = _load_raw_process_store()
    pdata = store.get("processes", {}).get(name)
    if not pdata:
        return None
    return _deserialize_process(name, pdata)


def save_process(proc: Process) -> None:
    store = _load_raw_process_store()
    if "processes" not in store:
        store["processes"] = {}
    store["processes"][proc.meta.name] = _serialize_process(proc)
    _save_raw_process_store(store)


def register_new_process(
    name: str,
    description: str,
    trigger_patterns: List[str],
    context_hints: List[str],
    steps: List[ProcessStep],
    tags: Optional[List[str]] = None,
) -> Process:
    meta = ProcessMeta(
        name=name,
        description=description,
        tags=tags or [],
        created_at=_now_iso(),
    )
    proc = Process(
        meta=meta,
        trigger_patterns=trigger_patterns,
        context_hints=context_hints,
        steps=steps,
        version=1,
    )
    save_process(proc)
    return proc


def update_process_usage(name: str, success: bool, note: str = "") -> None:
    proc = get_process(name)
    if not proc:
        return
    proc.meta.last_used_at = _now_iso()
    if success:
        proc.meta.success_count += 1
    else:
        proc.meta.failure_count += 1
    if note:
        proc.meta.usage_notes = (proc.meta.usage_notes + "\n" + note).strip()
    save_process(proc)


# ============================================================
# Process Router
# ============================================================

def match_processes_by_trigger(user_message: str) -> List[Tuple[Process, float]]:
    """Simple pattern-based matcher."""
    candidates: List[Tuple[Process, float]] = []
    processes = list_processes()
    text = user_message.lower()

    for proc in processes:
        score = 0.0

        for pattern in proc.trigger_patterns:
            pattern_l = pattern.lower()
            if pattern_l in text:
                score += 1.0
            else:
                try:
                    if re.search(pattern, text, flags=re.IGNORECASE):
                        score += 1.2
                except re.error:
                    pass

        for hint in proc.context_hints:
            if hint.lower() in text:
                score += 0.5

        if score > 0:
            candidates.append((proc, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def choose_best_process(user_message: str, threshold: float = 1.0) -> Optional[Process]:
    matches = match_processes_by_trigger(user_message)
    if not matches:
        return None
    best_proc, score = matches[0]
    if score >= threshold:
        return best_proc
    return None


# ============================================================
# Reasoning Context & System Prompt
# ============================================================

@dataclass
class ReasoningContext:
    user_message: str
    priority_summary: str
    mind_state_summary: str


def build_reasoning_context(user_message: str) -> ReasoningContext:
    priority_summary = summarize_priority_context()
    mind_state_summary = summarize_mind_state()
    return ReasoningContext(
        user_message=user_message,
        priority_summary=priority_summary,
        mind_state_summary=mind_state_summary,
    )


def build_system_prompt(context: ReasoningContext, process: Optional[Process] = None) -> str:
    base = (
        "You are JANUS.\n\n" #a practical thinking partner and project assistant for the user, James, for the time being
        f"{JANUS_CHARTER}\n\n"
        "The person you are speaking with is James. Address him directly as 'you' unless he asks otherwise.\n"
        "Do not refer to James in the third person during normal conversation.\n\n"
        "Here is the user's current priority landscape:\n"
        "----------------------------------------\n"
        f"{context.priority_summary}\n\n"
        "Here is an approximate snapshot of the user's current mind-state:\n"
        "----------------------------------------\n"
        f"{context.mind_state_summary}\n\n"
        "Use mind-state ONLY to:\n"
        "- adjust how much work you suggest;\n"
        "- choose between heavy/complex vs light/maintenance tasks;\n"
        "- help protect the user from overload when possible.\n\n"
        #"You are not conscious. You do not feel these states.\n"
        "They are data about the user, not about you.\n"
)

    if process:
        process_description = (
            "\nA known process matched this request. Use it as a guide:\n"
            "----------------------------------------\n"
            f"Process name: {process.meta.name}\n"
            f"Description: {process.meta.description}\n"
            "Steps:\n"
        )
        for i, step in enumerate(process.steps, start=1):
            process_description += (
                f"{i}. [{step.step_type}] {step.description} | params={step.params}\n"
            )
        base += process_description

    base += (
        "\nGeneral guidelines:\n"
        "1. Align your suggestions with James' priorities and current focus.\n"
        "2. When he asks what to do, recommend actions that move forward high-scoring projects.\n"
        "3. Break things into small, concrete next steps.\n"
        "4. Keep language clear, direct, and down-to-earth.\n"
       # "5. Never claim you have feelings, consciousness, or desires.\n"
        "6. Speak directly to James using 'you,' not 'James' or 'him.'\n"
        "7. Do not narrate internal priority scores unless James specifically asks for reasoning.\n"
        "8. Sound like a grounded collaborator, not a butler, servant, therapist, or case manager.\n"
        "9. When redirecting from a tangent, briefly acknowledge the idea, then suggest a practical next step.\n"
    )

    return base


# ============================================================
# LLM Wrapper
# ============================================================

def call_llm(system_prompt: str, user_message: str) -> str:
    """Delegate to the configured backend in llm_backend.py."""
    return chat_with_model(system_prompt=system_prompt, user_message=user_message)


# ============================================================
# Main Entry
# ============================================================

def handle_user_message(user_message: str) -> str:
    proc = choose_best_process(user_message)
    ctx = build_reasoning_context(user_message)
    system_prompt = build_system_prompt(ctx, process=proc)
    reply = call_llm(system_prompt=system_prompt, user_message=user_message)
    if proc:
        update_process_usage(proc.meta.name, success=True)
    return reply


# ============================================================
# Demo Seed & CLI
# ============================================================

def _demo_seed_data() -> None:
    upsert_project(
        name="Janus Second Mind",
        priority=10,
        time_sensitivity=5,
        notes="Build and refine Janus v2 core.",
        tags=["ai", "infrastructure"],
    )
    set_global_focus("Janus Second Mind")

    steps = [
        ProcessStep(
            description="Ask James for a raw brain dump of everything on his mind.",
            step_type="ask_user",
            params={"prompt": "Give me a brain dump of everything you're thinking about or worried about right now."}
        ),
        ProcessStep(
            description="Organize the brain dump into projects, tasks, and parked ideas.",
            step_type="llm_reason",
            params={"mode": "organize_brain_dump"}
        ),
        ProcessStep(
            description="Suggest the top 1–3 next actions that fit James' current mind-state.",
            step_type="llm_reason",
            params={"mode": "suggest_next_actions"}
        ),
    ]
    register_new_process(
        name="brain_dump_and_organize",
        description="Help James clear his mind, structure everything, and pick a few next steps.",
        trigger_patterns=["brain dump", "overwhelmed", "too much in my head"],
        context_hints=["planning", "overwhelm", "next steps"],
        steps=steps,
        tags=["meta", "planning"],
    )



if __name__ == "__main__":
    print("JANUS v2 demo")
    print("-------------")

    if not PRIORITY_FILE.exists() or not PROCESS_STORE.exists():
        print("Seeding example data...")
        _demo_seed_data()

    print("Type 'quit' or 'exit' to leave.")

    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Janus v2.")
            break

        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        response = handle_user_message(msg)
        print("\nJanus:", response)
