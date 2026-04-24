import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
from qa_engine import RAGEngine
from highlight import render_page_with_highlight
from config import DOCS_DIR

# --- Konfiguracja strony ---
st.set_page_config(
    page_title="PD Model — Walidacja Regulacyjna",
    page_icon="🏦",
    layout="wide",
)

# --- Funkcje pomocnicze ---
def run_ingestion_script():
    """Uruchamia proces indeksowania dokumentów."""
    with st.spinner("🚀 Indeksowanie dokumentów w toku... To może chwilę potrwać."):
        try:
            # Używamy sys.executable, aby mieć pewność użycia właściwego interpretera Pythona
            result = subprocess.run(
                [sys.executable, "ingestion.py"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                st.sidebar.success("✅ Indeksowanie zakończone pomyślnie!")
                # Wyświetlamy co zrobił skrypt wewnątrz expandera
                if result.stdout:
                    with st.sidebar.expander("Szczegóły operacji"):
                        st.code(result.stdout)
                st.rerun()
            else:
                st.sidebar.error("❌ Błąd podczas indeksowania")
                with st.sidebar.expander("Szczegóły błędu"):
                    st.code(result.stderr)
        except Exception as e:
            st.sidebar.error(f"💥 Błąd krytyczny podczas uruchamiania skryptu: {e}")

@st.cache_resource
def get_engine():
    return RAGEngine()

# Inicjalizacja silnika RAG
engine = get_engine()

# --- Główny tytuł ---
st.title("🏦 Model PD — Sprawdzanie zgodności regulacyjnej")

# --- SIDEBAR (Panel boczny) ---
st.sidebar.header("📂 Zarządzanie dokumentami")

# 1. Upload plików
uploaded_files = st.sidebar.file_uploader(
    "Wgraj nowe PDFy:", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Tworzymy folder docs, jeśli nie istnieje
        os.makedirs(DOCS_DIR, exist_ok=True)
        save_path = Path(DOCS_DIR) / uploaded_file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.info(f"Zapisano: {uploaded_file.name}")

# 2. Przycisk wyzwalający Ingestię
if st.sidebar.button("⚙️ Indeksuj dokumenty", use_container_width=True):
    run_ingestion_script()

st.sidebar.divider()

# 3. Lista załadowanych plików
st.sidebar.header("📚 Załadowane dokumenty")
docs_path = Path(DOCS_DIR)
if docs_path.exists():
    files = sorted(docs_path.glob("*.pdf"))
    if not files:
        st.sidebar.write("Brak plików w katalogu docs/")
    else:
        for f in files:
            st.sidebar.text(f"📄 {f.name}")
else:
    st.sidebar.warning("Brak katalogu docs/")

st.sidebar.divider()

# 4. Ustawienia modelu
st.sidebar.header("⚙️ Ustawienia")
top_k = st.sidebar.slider("Liczba fragmentów (top-k)", 3, 15, 8)
use_stream = st.sidebar.checkbox("Streaming odpowiedzi", value=True)

# --- GŁÓWNY UKŁAD (Main layout) ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("❓ Zadaj pytanie")

    # Szybkie sprawdzenie zmiennej
    column_name = st.text_input(
        "🔍 Szybkie sprawdzenie zmiennej:",
        placeholder="np. płeć, wiek, kod_pocztowy, dochód",
    )

    question = st.text_area(
        "Lub wpisz dowolne pytanie:",
        height=120,
        placeholder="Np. 'Jaki jest minimalny okres obserwacji do estymacji PD?'",
    )

    # Budowanie pytania na podstawie nazwy zmiennej
    if column_name and not question:
        question = (
            f"Czy mogę użyć zmiennej '{column_name}' w modelu oceny ryzyka kredytowego? "
            f"Sprawdź pod kątem CRR, wytycznych EBA, RODO "
            f"oraz AI Act. Oceń ryzyka i podaj konkretne artykuły."
        )

    if st.button("🔍 Sprawdź", type="primary", use_container_width=True):
        if not question:
            st.warning("Wpisz pytanie lub nazwę zmiennej.")
        else:
            with st.spinner("Przeszukuję regulacje..."):
                result = engine.ask(question, top_k=top_k, stream=use_stream)

            # Wyświetlanie odpowiedzi
            st.markdown("### 💬 Odpowiedź")
            if use_stream:
                answer_placeholder = st.empty()
                full_answer = ""
                for token in result["stream"]:
                    full_answer += token
                    answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
            else:
                st.markdown(result["answer"])

            # Wyświetlanie źródeł
            st.markdown("### 📎 Źródła")
            for s in result["sources"]:
                # W Qdrant 'score' to podobieństwo (zazwyczaj 0.0 - 1.0)
                similarity = s.get("score", 0)
                label = (
                    f"{'🟢' if similarity > 0.7 else '🟡' if similarity > 0.5 else '🔴'} "
                    f"{s['regulation']} — str. {s['page']} "
                    f"(trafność: {similarity:.0%})"
                )
                with st.expander(label):
                    st.text(s["text_preview"])
                    if st.button(
                        f"📄 Pokaż stronę {s['page']} w PDF",
                        key=f"show_{s['fragment_id']}_{s['page']}",
                    ):
                        st.session_state["pdf_file"] = s["file"]
                        st.session_state["pdf_page"] = s["page"]
                        # Używamy fragmentu tekstu do podświetlenia w PDF
                        st.session_state["pdf_highlight"] = s["text_preview"][:80]
                        st.rerun()

with col_right:
    st.subheader("📄 Podgląd dokumentu")

    if "pdf_file" in st.session_state:
        pdf_file = st.session_state["pdf_file"]
        pdf_page = st.session_state["pdf_page"]
        highlight_text = st.session_state.get("pdf_highlight")

        st.caption(f"**{pdf_file}** — strona {pdf_page}")

        # Nawigacja po stronach
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Poprzednia") and pdf_page > 1:
                st.session_state["pdf_page"] -= 1
                st.session_state["pdf_highlight"] = None
                st.rerun()
        with nav_col3:
            if st.button("Następna ➡️"):
                st.session_state["pdf_page"] += 1
                st.session_state["pdf_highlight"] = None
                st.rerun()

        # Renderowanie strony PDF
        try:
            img = render_page_with_highlight(
                pdf_file,
                st.session_state["pdf_page"],
                search_text=highlight_text,
            )
            if img:
                st.image(img, use_container_width=True)
            else:
                st.error(f"Błąd renderowania strony dla pliku: {pdf_file}")
        except Exception as e:
            st.error(f"Błąd: {e}")
    else:
        st.info(
            "Kliknij **'Pokaż stronę w PDF'** przy źródle, aby zobaczyć dokument."
        )