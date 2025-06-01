# -*- coding: utf-8 -*-
"""
Enhanced Gradio UI (English Version)
====================================
Front‑end for *model.py* with fully‑translated labels, tooltips and help‑texts plus a cleaner
layout that uses Tabs instead of long scroll sections.

Key changes
-----------
* **Language** – all Chinese annotations → English.
* **Layout** – compact two‑column design + Tabs (Results · System Status · Stats).
* **Styling** – Tailwind‑like palette via custom CSS.
* **Helper buttons** – tooltips replaced by `elem_id` hover popovers.
* **Live word counter** – updates below the textbox.
"""

from datetime import datetime
import time
import gradio as gr
import sys

try:
    from model import (
        master_grammar_check,
        MODEL_OPTIONS,
        get_language_tool_instance,
        get_spell_checker_instance,
        test_bootkass_corrections,
        loaded_models,
    )
except ImportError:
    print("❗ Cannot import model.py – make sure it exists in the same directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Example sentences (same logic – just English tool‑tips)
# -----------------------------------------------------------------------------

EXAMPLES = {
    "Spelling (basic)": "I has a bad grammer and need to improve my writting skils.",
    "Bootkass case": "The students in the classroom are reading their bootkass carefully.",
    "Grammar": "He don't like water but he swim very good yesterday.",
    "Punctuation": "this is a sentence without proper punctuation what do you think",
    "Complex": "Their going to there house with they're friends for a party.",
    "Tense": "I have went to the store yesterday and buyed some foods.",
    "Subject‑verb": "The group of students are very smart and they has many talents.",
    "Mixed": "Me and him was walking to the store yesterday, we dont have many monies.",
    "Common": "The studients are in the clasroom with there techer.",
    "Plural": "I see many childs playing in the parks with there toys.",
}

# -----------------------------------------------------------------------------
# Runtime stats
# -----------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = datetime.now()
        self.total = 0
        self.ok = 0
        self.model_hits = {m: 0 for m in MODEL_OPTIONS}
        self.spell = self.grammar = 0

    # -----------------------------------------------------
    def push(self, res: dict, model_key: str):
        self.total += 1
        if "error" not in res:
            self.ok += 1
            self.model_hits[model_key] += 1
            self.spell += len(res.get("spell_suggestions", []))
            self.grammar += len(res.get("grammar_errors", []))

    # -----------------------------------------------------
    def markdown(self) -> str:
        if not self.total:
            return "*No statistics yet.*"
        uptime = str(datetime.now() - self.t0).split(".")[0]
        hit_lines = "\n".join([f"- **{m}**: {n}" for m, n in self.model_hits.items() if n]) or "- *None*"
        return f"""
### 📊 Runtime Statistics
* Sessions: **{self.total}** (success {self.ok}/{self.total})
* Uptime: **{uptime}**
* Models used:\n{hit_lines}
* Spelling issues found: **{self.spell}**
* Grammar issues found: **{self.grammar}**
"""

STATS = Stats()

# -----------------------------------------------------------------------------
# CSS (light tweaks)
# -----------------------------------------------------------------------------

CSS = """
.gradio-container {max-width: 1200px;margin:auto;font-family:Inter,Segoe UI,sans-serif}
.title {text-align:center;padding:24px;border-radius:12px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;margin-bottom:12px}
.title h1{font-size:2.2rem;margin:0}
.btn-example button{font-size:.85rem;padding:6px 10px}
.diff-added{background:#e6ffed;color:#22863a;padding:2px 4px;border-radius:4px;font-weight:600}
.diff-removed{background:#ffeef0;color:#b31d28;text-decoration:line-through;padding:2px 4px;border-radius:4px}
"""

# -----------------------------------------------------------------------------
# Helper   –  format result blocks
# -----------------------------------------------------------------------------

def ui_blocks(res: dict):
    if "error" in res:
        return f"❌ **Error**: {res['error']}", "", ""

    # ---- Main summary
    summary = (
        "✅ *No issues found*" if not res.get("stats", {}).get("transformer_changes")
        else f"✅ **Corrected**\n\n```\n{res['primary_correction']}\n```"
    )

    # ---- Spelling list
    spell_cards = []
    for itm in res.get("spell_suggestions", [])[:5]:
        conf = itm.get("confidence", "low")
        spell_cards.append(f"- **{itm['word']} → {itm['correction']}** ({conf})")
    spelling_md = "\n".join(spell_cards) or "*No spelling issues*"

    # ---- Grammar list
    grammar_cards = []
    for itm in res.get("grammar_errors", [])[:5]:
        sev = itm.get("severity", "Medium")
        grammar_cards.append(f"- **{itm['message']}** → {itm['replacement']} ({sev})")
    grammar_md = "\n".join(grammar_cards) or "*No grammar issues*"

    # ---- Diff
    diff_html = "".join(
        f"<span class='diff-{('removed' if tag=='REMOVED' else 'added') if tag in ('REMOVED','ADDED') else ''}'>{txt}</span>"
        if tag else txt for txt, tag in res.get("diff_highlight", [])
    ) or "<i>No changes</i>"
    return summary, spelling_md, grammar_md, diff_html

# -----------------------------------------------------------------------------
# Core callback
# -----------------------------------------------------------------------------

def run_check(text, model_key, spell_ck, grammar_ck):
    if not text.strip():
        return "Please enter some text.", "", "", "", STATS.markdown()

    t0 = time.time()
    res = master_grammar_check(
        text_to_check=text,
        selected_model_key=model_key,
        use_language_tool=grammar_ck,
        use_pyspellchecker=spell_ck,
    )
    res["latency"] = f"{time.time()-t0:.2f}s"
    STATS.push(res, model_key)

    main, spell_md, grammar_md, diff_html = ui_blocks(res)
    return main, spell_md, grammar_md, diff_html, STATS.markdown()

# -----------------------------------------------------------------------------
# System status drawer
# -----------------------------------------------------------------------------

def status_panel():
    lt = "Available" if get_language_tool_instance() else "Unavailable"
    sc = "Available" if get_spell_checker_instance() else "Unavailable"
    gpu = "CUDA" if __import__("torch").cuda.is_available() else "CPU"
    models = len(loaded_models)
    return f"""
**LanguageTool**: {lt}   |   **SpellChecker**: {sc}   |   **Backend**: {gpu}   |   **Loaded models**: {models}
"""

# -----------------------------------------------------------------------------
# Build Gradio interface
# -----------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(css=CSS, title="AI Grammar Checker (EN)") as gui:
        gr.HTML("""<div class='title'><h1>AI Grammar Checker</h1><p>Transformer‑based grammar & spelling correction with resilient fallbacks.</p></div>""")

        with gr.Row():
            # ---------- Left column ----------
            with gr.Column(scale=3):
                txt = gr.Textbox(label="Input text", lines=8, placeholder="Paste or type English text here …")
                live = gr.Markdown("Word count: 0")

                with gr.Row():
                    model_dd = gr.Dropdown(list(MODEL_OPTIONS.keys()), value=list(MODEL_OPTIONS)[0], label="Model")
                    spell_cb = gr.Checkbox(True, label="Spell‑checker")
                    gram_cb = gr.Checkbox(True, label="LanguageTool")

                with gr.Row():
                    run_btn = gr.Button("Check", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Markdown("### Examples")
                for name, sent in EXAMPLES.items():
                    gr.Button(name, elem_classes=["btn-example"]).click(lambda s=sent: s, outputs=txt)

            # ---------- Right column ----------
            with gr.Column(scale=2):
                with gr.Tab("Results"):
                    summary = gr.Markdown()
                    gr.Markdown("#### Spelling")
                    spell_md = gr.Markdown()
                    gr.Markdown("#### Grammar")
                    grammar_md = gr.Markdown()
                    gr.Markdown("#### Diff")
                    diff_html = gr.HTML()

                with gr.Tab("System"):
                    sys_md = gr.Markdown(status_panel())

                with gr.Tab("Stats"):
                    stats_md = gr.Markdown(STATS.markdown())

        # ---------- Bindings ----------
        txt.change(lambda t: f"Word count: {len(t.split())}", txt, live)
        run_btn.click(run_check, [txt, model_dd, spell_cb, gram_cb], [summary, spell_md, grammar_md, diff_html, stats_md])
        clear_btn.click(lambda: ("", "Word count: 0"), outputs=[txt, live])

        # Extra buttons (footer)
        with gr.Row():
            gr.Button("Bootkass Test").click(lambda: test_bootkass_corrections() or "Bootkass test completed! Check console.", outputs=summary)
            gr.Button("Reset Stats").click(lambda: (STATS.reset() or "Statistics reset."), outputs=summary).then(lambda: STATS.markdown(), outputs=stats_md)
    return gui


if __name__ == "__main__":
    build_ui().launch(show_error=True, server_name="0.0.0.0", server_port=7860, inbrowser=True)
