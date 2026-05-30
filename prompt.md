I have a Streamlit RAG app (Streamlit 1.49.1) with two unsolved CSS/layout problems.

## Problems

1. **Sidebar not visible**: The sidebar content is not appearing visually. `st.set_page_config(initial_sidebar_state="expanded")` is set. The `with st.sidebar:` block has file uploader and buttons. The sidebar either:
   - Is collapsed and the collapse button is not visible/clickable
   - Has the same background color as the main content (`#0d0d0d` via config.toml `secondaryBackgroundColor`)
   - Is hidden by some CSS conflict

2. **Two-tone chat panel**: The main chat area shows two different background colors — some elements (like text area, form container) appear with a different dark shade than the uniform `#0d0d0d` background.

## Current approach

I'm using `.streamlit/config.toml`:
```toml
[theme]
backgroundColor = "#0d0d0d"
secondaryBackgroundColor = "#0a0a0a"
primaryColor = "#e0e0e0"
textColor = "#e0e0e0"
```

And a large CSS block injected via `st.markdown("<style>...</style>", unsafe_allow_html=True)` that overrides many Streamlit component styles with `!important`.

## What I need

- Sidebar with subtly distinct background (e.g., `#0a0a0a`) and visible right border (`1px solid #222`)
- Uniform `#0d0d0d` background everywhere else (chat area, form, messages)
- Sidebar collapse button visible and functional (small, subtle styling only)
- "Build vector store" button and file uploader in sidebar
- Input form at bottom with text area and send button

Can you give me a clean working solution?
