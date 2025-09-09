"""
Streamlit GUI for Quran Tafsir Semantic Search
A modern web-based interface for searching through Quran commentary (tafsir)
"""

# Fix SQLite version compatibility for ChromaDB
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import sqlite3
import chromadb
from chromadb.utils import embedding_functions

# Constants
UTHMANI_DB_PATH = "quran-data/quran/uthmani.db"
AUDIO_DB_PATH = "quran-data/recitation/ayah-recitation-khalifa-al-tunaiji-murattal-hafs-958.db"
METADATA_DB_PATH = "quran-data/metadata/quran-metadata-surah-name.sqlite"


@st.cache_resource
def load_collection():
    """Load ChromaDB collection with caching for better performance"""
    return get_existing_collection()


def create_tafsir_database_from_sqlite():
    """
    Create ChromaDB collection from SQLite data if needed.
    """
    print("Creating tafsir database from SQLite...")
    
    # Connect to SQLite database
    conn = sqlite3.connect("quran-data/tafseer/abridged-explanation-of-the-quran.db")
    cursor = conn.cursor()
    
    # Fetch all tafsir data
    cursor.execute("SELECT ayah_key, text FROM tafsir")
    rows = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} tafsir entries")
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for ayah_key, text in rows:
        # Clean and prepare text
        cleaned_text = text.strip()
        if not cleaned_text:  # Skip empty texts
            continue
            
        documents.append(cleaned_text)
        
        # Parse ayah_key (e.g., "1:5" -> surah=1, verse=5)
        surah, verse = ayah_key.split(':')
        metadatas.append({
            'ayah_key': ayah_key,
            'surah': int(surah),
            'verse': int(verse),
            'text_length': len(cleaned_text)
        })
        
        # Use ayah_key as unique ID
        ids.append(f"tafsir_{ayah_key}")
    
    print(f"Prepared {len(documents)} documents for embedding")
    
    # Create ChromaDB persistent client
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Use multilingual embedding model for better Arabic/English support
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Create collection (delete if exists)
    try:
        client.delete_collection("quran_tafsir")
    except Exception:
        pass  # Collection doesn't exist
    
    collection = client.create_collection(
        name="quran_tafsir",
        embedding_function=sentence_transformer_ef,
        metadata={"description": "Abridged explanation of the Quran tafsir"}
    )
    
    # Add documents in batches (ChromaDB has limits)
    batch_size = 1000
    print("Adding documents to ChromaDB...")
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
    
    print(f"Successfully loaded {len(documents)} tafsir entries into ChromaDB!")
    return collection


def get_existing_collection():
    """
    Get existing ChromaDB collection without recreating it.
    
    Returns:
        Existing ChromaDB collection
    """
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Use the same embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    try:
        collection = client.get_collection(
            name="quran_tafsir",
            embedding_function=sentence_transformer_ef
        )
        print(f"Connected to existing collection with {collection.count()} documents")
        return collection
    except Exception as e:
        print(f"Error loading existing collection: {e}")
        print("Rebuilding collection from SQLite data...")
        # Rebuild the collection from SQLite
        return create_tafsir_database_from_sqlite()


def search_tafsir(collection, query: str, n_results: int = 5, surah_filter=None):
    """
    Search tafsir explanations and return ayah_keys with explanations.
    """
    where_clause = None
    if surah_filter:
        where_clause = {"surah": surah_filter}
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )
    
    return results


def fetch_qurtubi_tafsir(ayah_key: str) -> str:
    """Fetch Qurtubi tafsir for a given ayah_key from ar-tafseer-al-qurtubi.db"""
    db_path = "quran-data/tafseer/ar-tafseer-al-qurtubi.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM tafsir WHERE ayah_key = ? LIMIT 1", (ayah_key,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        return "No Qurtubi tafsir found for this ayah."
    except Exception as e:
        return f"Error loading Qurtubi tafsir: {e}"


@st.cache_data
def fetch_ayah_text(ayah_key: str) -> str:
    """Fetch Arabic text of an ayah from the database with caching"""
    try:
        conn = sqlite3.connect(UTHMANI_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM verses WHERE verse_key = ?", (ayah_key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "[Ayah not found]"
    except Exception as e:
        return f"[Error loading ayah: {e}]"


@st.cache_data
def fetch_audio_url(surah_num: int, verse_num: int) -> str:
    """Build audio URL for an ayah from the recitation database with caching"""
    try:
        # Build the audio URL directly using the pattern
        audio_url = f"https://audio-cdn.tarteel.ai/quran/khalifaAlTunaiji/{surah_num:03d}{verse_num:03d}.mp3"
        return audio_url
    except Exception as e:
        print(f"Error building audio URL: {e}")
        return ""


@st.cache_data
def fetch_surah_name(surah_num: int) -> str:
    """Fetch surah name from metadata database with caching"""
    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM chapters WHERE id = ?", (surah_num,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else f"Surah {surah_num}"
    except Exception as e:
        print(f"Error fetching surah name: {e}")
        return f"Surah {surah_num}"


def display_search_result(result_doc, result_meta, result_distance, idx):
    """Display a single search result in a styled container"""
    ayah_key = result_meta['ayah_key']
    surah_num = result_meta['surah']
    verse_num = result_meta['verse']
    
    # Calculate similarity percentage
    similarity_percent = (1 - result_distance) * 100 if result_distance is not None else 0
    
    # Fetch additional data
    arabic_text = fetch_ayah_text(ayah_key)
    audio_url = fetch_audio_url(surah_num, verse_num)
    surah_name = fetch_surah_name(surah_num)
    
    # Create result container with custom styling
    with st.container():
        # Header with surah name, verse number, and similarity score (single line)
        verse_url = f"https://quran.com/{surah_num}?startingVerse={verse_num}"
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: nowrap;'>
                <div style='flex: 1; min-width: 0;'>
                    <a href='{verse_url}' target='_blank'><strong>{surah_name} : {verse_num}</strong></a>
                </div>
                <div style='font-weight: bold; white-space: nowrap; margin-left: 10px;'>
                    Similarity: {similarity_percent:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Arabic text (RTL) - Use Streamlit container with responsive text
        arabic_container = st.container()
        with arabic_container:
            st.markdown(
                f'<p dir="rtl" style="font-family: \'Amiri\', \'Traditional Arabic\', serif; font-size: 1.3em; line-height: 1.8; margin: 15px 0; text-align: right;">{arabic_text}</p>', 
                unsafe_allow_html=True
            )
        
        # Audio player
        if audio_url:
            st.audio(audio_url, format="audio/mp3")
        
        # Tafsir text - use Streamlit's native text display
        st.markdown(result_doc)
        
        # Qurtubi tafsir in an expander
        with st.expander("Qurtubi Tafsir", expanded=False):
            qurtubi_tafsir = fetch_qurtubi_tafsir(ayah_key)
            st.markdown(f'<div dir="rtl" style="font-family: \'Amiri\', \'Traditional Arabic\', serif;">{qurtubi_tafsir}</div>', unsafe_allow_html=True)
        
        # Space between results using Streamlit's native spacing
        st.markdown("")
        st.markdown("")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Search the Quran",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Minimal CSS - only essential RTL support not provided by Streamlit
    st.markdown("""
    <style>
    /* Hide the "Press Enter to apply" message for cleaner UX */
    .stTextInput > div > div > div > div {
        display: none !important;
    }
    
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Simple header
    st.title("Search the Quran")
    st.markdown("Search through Quranic commentary using natural language")
    
    # Load collection
    if 'collection' not in st.session_state:
        with st.spinner("Loading search database..."):
            st.session_state.collection = load_collection()
    
    collection = st.session_state.collection
    
    # Main search interface - centered and prominent, only Semantic Search

    # Main search box - prominent and full width (first input)
    
    # Use session state for query value
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    # Create search input with submit button for mobile-friendly interaction
    search_col1, search_col2 = st.columns([4, 1])
    
    with search_col1:
        query = st.text_input(
            "Search the Quranic Commentary",
            value=st.session_state.search_query,
            placeholder="Ask about Quranic themes or concepts...",
            help="Search the Quran's tafsir (commentary) using natural language. Find explanations, themes, and insights from the Quran.",
            label_visibility="collapsed",
            max_chars=200,
            key="search_input"
        )
    
    with search_col2:
        search_pressed = st.button("Search", use_container_width=True)
    
    # Update session state and trigger search on button press or text input
    if search_pressed and query.strip():
        st.session_state.search_query = query.strip()
        st.rerun()
    elif query != st.session_state.search_query:
        st.session_state.search_query = query

    # Filter controls in a collapsible expander (hidden by default)
    with st.expander("Advanced Search Options", expanded=False):
        filter_cols = st.columns([3, 2])
        with filter_cols[0]:
            # Get all surahs from database for the dropdown
            conn = sqlite3.connect(METADATA_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name_simple FROM chapters ORDER BY id")
            all_surahs = cursor.fetchall()
            conn.close()
            
            surah_options = ["All Surahs"] + [f"{surah_id}: {name}" for surah_id, name in all_surahs]
            selected_surah = st.selectbox("Select Surah (optional)", surah_options)
            surah_filter = None
            if selected_surah != "All Surahs":
                surah_filter = int(selected_surah.split(":")[0])
        with filter_cols[1]:
            similarity_threshold = st.slider(
                "Minimum Similarity (%)",
                min_value=0,
                max_value=100,
                value=35,
                help="Only show results with similarity greater than or equal to this value."
            )

    # Show what types of queries the app can handle
    st.markdown("""
    *Try searching for themes like "mercy and compassion in times of difficulty" • 
    "what the Quran teaches about dealing with injustice" • 
    "stories and lessons from Prophet Abraham" • 
    "guidance on maintaining faith during trials"*
    """)

    n_results = 50

    # Perform search (triggered by button press or text input change)
    current_query = st.session_state.search_query.strip()
    if current_query:
        with st.spinner("Searching..."):
            results = search_tafsir(collection, current_query, n_results, surah_filter)
            if results['documents'][0]:
                found = False
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = (1 - distance) * 100
                    if similarity >= similarity_threshold:
                        display_search_result(doc, metadata, distance, i)
                        found = True
                if not found:
                    st.warning("No results found above the selected similarity threshold.")
            else:
                st.warning("No results found. Try a different search query.")

    # Footer
    st.markdown("---")
    
    # Data Sources Section
    st.markdown("### Data Sources")
    st.markdown("""
    This application uses authentic Islamic resources to provide accurate Quranic content:
    
    **Quran Text (Arabic)**  
    Digital Khatt Script - [Tarteel Resources](https://qul.tarteel.ai/resources/quran-script/56)
    
    **English Translation**  
    Saheeh International Translation - [Tarteel](https://qul.tarteel.ai/resources/translation/193) | [Wikipedia](https://en.wikipedia.org/wiki/Sahih_International)
    
    **Arabic Tafseer (Commentary)**  
    Scholarly Commentary - [Tarteel Resources](https://qul.tarteel.ai/resources/tafsir/23)
    
    All content is provided for educational and spiritual enrichment purposes.
    """)
    
    st.markdown("---")
    st.markdown(
        """
        **Developed by [National AI Powered Systems (NAPS)](https://nationalai.net/en/)**  
        *Libya's leading AI & data transformation partner*
        """, 
        unsafe_allow_html=False
    )


if __name__ == "__main__":
    main()
