import os
import json
import streamlit as st

OUTPUTS_ROOT = "./outputs"

# Get all available sessions (subfolders)
def get_sessions(outputs_root):
    subs = [
        d for d in os.listdir(outputs_root)
        if os.path.isdir(os.path.join(outputs_root, d))
        and os.path.isfile(os.path.join(outputs_root, d, "results.jsonl"))
    ]
    # sort by date DESC, latest first
    return sorted(subs, reverse=True)

# Load results.jsonl file for one session
def load_results(session_folder):
    results_path = os.path.join(OUTPUTS_ROOT, session_folder, "results.jsonl")
    with open(results_path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]

# --- Streamlit UI ---

st.set_page_config(page_title="SuzieAI : Evaluation automatique à échelle", page_icon="📊", layout="wide")

st.title("📊 SuzieAI : Evaluation automatique à échelle")

# ... existing code ...

sessions = get_sessions(OUTPUTS_ROOT)
if not sessions:
    st.warning("No sessions found in ./outputs/")
    st.stop()

# --- Compute summary metrics over all sessions ---
session_for_metrics = sessions[0] if sessions else None
results_for_avg = load_results(session_for_metrics) if session_for_metrics else []
n_results = len(results_for_avg)

if n_results > 0:
    # Collect metrics
    all_cos = [float(ex.get("cosine_similarity", 0.0)) for ex in results_for_avg]
    all_bleu = [float(ex.get("bleu", 0.0)) for ex in results_for_avg]
    all_rouge1r = []
    for ex in results_for_avg:
        rouge = ex.get("rouge", {})
        rouge_1 = rouge.get("rouge-1", {}) if isinstance(rouge, dict) else {}
        all_rouge1r.append(float(rouge_1.get("r", 0.0)))
    avg_cos = sum(all_cos) / n_results if n_results else 0.0
    avg_bleu = sum(all_bleu) / n_results if n_results else 0.0
    avg_rouge1r = sum(all_rouge1r) / n_results if n_results else 0.0

    # Judge rates
    count_labels = {"correcte": 0, "partielle": 0, "incorrecte": 0}
    for ex in results_for_avg:
        lbl = ex.get('judge', {}).get('label', None)
        if lbl in count_labels: count_labels[lbl] += 1
    pct_total = lambda c: 100.0 * c / n_results if n_results else 0.0
    pct_corr = pct_total(count_labels["correcte"])
    pct_part = pct_total(count_labels["partielle"])
    pct_incorr = pct_total(count_labels["incorrecte"])
    label_summary_html = (
        f'<span style="color:green;font-weight:bold">{pct_corr:.0f} %</span> / '
        f'<span style="color:orange;font-weight:bold">{pct_part:.0f} %</span> / '
        f'<span style="color:red;font-weight:bold">{pct_incorr:.0f} %</span>'
    )
else:
    avg_cos = avg_bleu = avg_rouge1r = 0.0
    label_summary_html = "---"

# --- Display session averages above session select ---
sum0, sum1, sum2, sum3 = st.columns(4)
with sum0:
    st.markdown("**Réponses LLM :**<br>" + label_summary_html, unsafe_allow_html=True, help="Pourcentage de réponses jugées correcte (vert), partielle (orange), incorrecte (rouge) par le LLM sur l'ensemble de la session")
with sum1:
    st.metric("Cosine Sim. (moy)", f"{avg_cos*100:.1f} %", help="Moyenne de la similarité sémantique entre chaque réponse générée et attendue (plus proche de 100% = plus similaire en sens)")
with sum2:
    st.metric("BLEU (moy)", f"{avg_bleu*100:.1f} %", help="Moyenne de la similarité de phrases (corrélation par mots, 100% = identique)")
with sum3:
    st.metric("ROUGE-1 Recall (moy)", f"{avg_rouge1r*100:.1f} %", help="Moyenne du taux de correspondance des mots du texte attendu trouvés dans la réponse générée")

# --- Continue as before: show session picker, example details, etc ---
session = st.selectbox(
    "Sélectionnez une session (par date):", 
    options=sessions,
    format_func=lambda x: x
)

results = load_results(session)
total = len(results)

# ... rest of code unchanged ...

if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

col1, col2, col3 = st.columns([1, 9, 1])

with col1:
    if st.button("⬅️", disabled=(st.session_state.current_idx == 0)):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
with col3:
    if st.button("➡️", disabled=(st.session_state.current_idx == total-1)):
        st.session_state.current_idx = min(total-1, st.session_state.current_idx + 1)

st.write(f"**Exemple {st.session_state.current_idx+1} / {total}**")

ex = results[st.session_state.current_idx]

st.markdown("#### 📝 Question")
st.code(ex["input"], language="markdown")

st.markdown("#### 🎯 Réponse attendue")
st.code(ex["output_expected"], language="markdown")

# --- Extract and format metrics ---
cos_sim = ex.get("cosine_similarity", 0.0)
bleu = ex.get("bleu", 0.0)
# Get ROUGE-1 recall (r) from ["rouge"]["rouge-1"]["r"]
rouge1_r = 0.0
rouge = ex.get("rouge", {})
if isinstance(rouge, dict):
    rouge_1 = rouge.get("rouge-1", {})
    rouge1_r = float(rouge_1.get("r", 0.0))

cos_sim_pct = f"{cos_sim*100:.1f} %"
bleu_pct = f"{bleu*100:.1f} %"
rouge1_r_pct = f"{rouge1_r*100:.1f} %"

# Display judge label with color
label = ex.get('judge', {}).get('label', '-')
color = {"correcte": "green", "partielle": "orange", "incorrecte": "red"}.get(label, "gray")
label_html = f'<span style="color:{color}; font-weight:bold; font-size:2.0em;">{label}</span>'

# 4 columns: 1st = judge label, then metrics
c0, c1, c2, c3 = st.columns(4)
with c0:
    st.markdown("**Réponse évaluée comme :**<br>" + label_html, unsafe_allow_html=True, help="Evaluation d'un LLM de la réponse générée par rapport à la réponse attendue (à mitiger avec son propre jugement)")
with c1:
    st.metric(label="Cosine Similarity (%)", value=cos_sim_pct, help="Proximité sémantique (plus proche de 100% = plus similaire en sens)")
with c2:
    st.metric(label="BLEU (%)", value=bleu_pct, help="Similarité de phrases (corrélation par mots, 100% = identique)")
with c3:
    st.metric(label="ROUGE-1 Recall (%)", value=rouge1_r_pct, help="Taux de correspondance des mots du texte attendu trouvés dans la réponse générée, (100% = identique)")

st.markdown("#### 🤖 Réponse générée")
st.code(ex["output_inference"], language="markdown")

if "judge" in ex:
    st.markdown("**Justification de l'évaluation LLM :**")
    st.write(ex['judge'].get('justification',''))

st.markdown("---")
st.caption(f"Session: `{session}`  |  Fichier: `results.jsonl`")