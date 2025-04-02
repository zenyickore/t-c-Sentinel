import streamlit as st
import os
import tempfile
import base64
from document_processor import DocumentProcessor
from comparison_engine import ComparisonEngine

# Set page configuration
st.set_page_config(
    page_title="T&C Sentinel - Contract Comparison Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .category-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-container {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-indicator {
        font-weight: bold;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        display: inline-block;
    }
    .status-warning {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .status-ok {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .status-error {
        background-color: #FEE2E2;
        color: #B91C1C;
    }
    .severity-minor {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
    }
    .severity-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
    }
    .severity-major {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
    }
    .action-needed {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .acceptable {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state variables if they don't exist
    if 'has_master_document' not in st.session_state:
        st.session_state.has_master_document = False
    if 'master_filename' not in st.session_state:
        st.session_state.master_filename = None
    if 'master_path' not in st.session_state:
        st.session_state.master_path = None
    
    # Header
    st.markdown('<h1 class="main-header">T&C Sentinel</h1>', unsafe_allow_html=True)
    st.markdown("### Compare Terms & Conditions documents to identify discrepancies and risks")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Document processing settings
        st.subheader("Document Processing")
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100, 
                              help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=200, step=50,
                                 help="Overlap between chunks to maintain context")
        
        # Comparison settings
        st.subheader("Comparison Settings")
        retrieval_k = st.slider("Retrieval Count", min_value=1, max_value=10, value=5, step=1,
                               help="Number of relevant chunks to retrieve for each category")
        
        # Advanced settings
        st.subheader("Advanced Settings")
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                               help="Higher values make output more random, lower values more deterministic")
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            options=["openai", "saul"],
            index=0,
            help="Select the model to use for analysis (OpenAI GPT-4 or Saul-Instruct-v1)"
        )
        
        # Master document management
        st.subheader("Master Document Management")
        if st.session_state.has_master_document:
            st.success(f"Current master: {st.session_state.master_filename}")
            if st.button("Clear Master Document"):
                # Initialize comparison engine
                comparison_engine = ComparisonEngine(temperature=temperature, model_type=model_type)
                # Delete the master vector database
                comparison_engine.delete_vector_db("master")
                # Reset session state
                st.session_state.has_master_document = False
                st.session_state.master_filename = None
                st.session_state.master_path = None
                st.success("Master document cleared")
                st.rerun()
        else:
            st.info("No master document set")
        
        # About section
        st.subheader("About")
        st.markdown("""
        **T&C Sentinel** helps legal teams identify discrepancies between master and client T&C documents.
        
        Built with:
        - Streamlit
        - LangChain
        - OpenAI GPT-4
        - ChromaDB
        """)
    
    # Initialize processors with user settings
    doc_processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    comparison_engine = ComparisonEngine(
        temperature=temperature,
        model_type=model_type
    )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Master Document")
        master_file = st.file_uploader("Upload master T&C document", type=["pdf"], key="master")
        
        if master_file:
            st.success(f"Uploaded: {master_file.name}")
            
            # Display document metadata if available
            with st.expander("Document Metadata"):
                # Create a temporary file to process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(master_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Extract metadata
                metadata = doc_processor.extract_metadata(tmp_path)
                
                if metadata:
                    for key, value in metadata.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write("No metadata available")
            
            # Option to set as master document
            if not st.session_state.has_master_document or st.session_state.master_filename != master_file.name:
                if st.button("Set as Master Document"):
                    with st.spinner("Processing master document..."):
                        # Process document
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(master_file.getvalue())
                            master_path = tmp_file.name
                            
                            # Save a permanent copy
                            os.makedirs("./master_documents", exist_ok=True)
                            permanent_path = f"./master_documents/{master_file.name}"
                            with open(permanent_path, 'wb') as f:
                                f.write(master_file.getvalue())
                        
                        # Process document
                        master_data = doc_processor.process_document(master_path)
                        
                        # Create vector database with persistence
                        master_db = comparison_engine.create_vector_db(
                            document_chunks=master_data["chunks"],
                            namespace="master",
                            persist=True
                        )
                        
                        # Update session state
                        st.session_state.has_master_document = True
                        st.session_state.master_filename = master_file.name
                        st.session_state.master_path = permanent_path
                        
                        # Clean up temporary file
                        os.unlink(master_path)
                    
                    st.success(f"Set {master_file.name} as master document")
                    st.rerun()
    
    with col2:
        st.header("Client Document")
        client_file = st.file_uploader("Upload client T&C document", type=["pdf"], key="client")
        
        if client_file:
            st.success(f"Uploaded: {client_file.name}")
            
            # Display document metadata if available
            with st.expander("Document Metadata"):
                # Create a temporary file to process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(client_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Extract metadata
                metadata = doc_processor.extract_metadata(tmp_path)
                
                if metadata:
                    for key, value in metadata.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write("No metadata available")
    
    # Comparison section
    st.markdown("---")
    st.header("Document Comparison")
    
    # Check if we have a master document (either from session or newly uploaded)
    has_master = st.session_state.has_master_document or master_file is not None
    
    if has_master and client_file:
        if st.button("Compare Documents", type="primary"):
            with st.spinner("Processing documents and performing comparison..."):
                # Process master document if needed
                if st.session_state.has_master_document:
                    # Load existing master vector database
                    master_db = comparison_engine.load_master_vector_db()
                    master_path = st.session_state.master_path
                    master_filename = st.session_state.master_filename
                else:
                    # Process new master document
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(master_file.getvalue())
                        master_path = tmp_file.name
                    
                    # Process document
                    master_data = doc_processor.process_document(master_path)
                    
                    # Create vector database
                    master_db = comparison_engine.create_vector_db(
                        document_chunks=master_data["chunks"],
                        namespace="master"
                    )
                    
                    master_filename = master_file.name
                
                # Process client document
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(client_file.getvalue())
                    client_path = tmp_file.name
                
                # Process client document
                client_data = doc_processor.process_document(client_path)
                
                # Create client vector database
                client_db = comparison_engine.create_vector_db(
                    document_chunks=client_data["chunks"],
                    namespace="client"
                )
                
                # Set up retrievers
                master_retriever = comparison_engine.setup_retriever(
                    vector_db=master_db,
                    k=retrieval_k
                )
                
                client_retriever = comparison_engine.setup_retriever(
                    vector_db=client_db,
                    k=retrieval_k
                )
                
                # Perform comparison
                comparison_results = comparison_engine.compare_documents(
                    master_retriever=master_retriever,
                    client_retriever=client_retriever
                )
                
                # Store results in session state
                st.session_state.comparison_results = comparison_results
                st.session_state.master_filename = master_filename
                st.session_state.client_filename = client_file.name
                st.session_state.client_path = client_path
                
                # Don't delete master_path if it's the persistent one
                if not st.session_state.has_master_document and master_file:
                    os.unlink(master_path)
            
            st.success("Comparison completed!")
    else:
        if not has_master:
            st.info("Please upload or set a master document")
        if not client_file:
            st.info("Please upload a client document")
    
    # Display results if available
    if "comparison_results" in st.session_state:
        st.markdown("---")
        st.header("Comparison Results")
        st.markdown(f"**Master Document:** {st.session_state.master_filename}")
        st.markdown(f"**Client Document:** {st.session_state.client_filename}")
        
        # Overall summary
        st.subheader("Executive Summary")
        st.markdown(st.session_state.comparison_results["overall_summary"])
        
        # Create tabs for categories
        categories = comparison_engine.comparison_categories
        tabs = st.tabs(categories)
        
        # Display results for each category
        for i, category in enumerate(categories):
            with tabs[i]:
                result = st.session_state.comparison_results[category]
                
                # Status indicators
                status_col1, status_col2, status_col3 = st.columns(3)
                
                with status_col1:
                    status_class = "status-warning" if result["has_discrepancies"] else "status-ok"
                    status_text = "Discrepancies Found" if result["has_discrepancies"] else "No Discrepancies"
                    st.markdown(f'<div class="status-indicator {status_class}">{status_text}</div>', unsafe_allow_html=True)
                
                with status_col2:
                    status_class = "status-warning" if result["has_contradictions"] else "status-ok"
                    status_text = "Contradictions Found" if result["has_contradictions"] else "No Contradictions"
                    st.markdown(f'<div class="status-indicator {status_class}">{status_text}</div>', unsafe_allow_html=True)
                
                with status_col3:
                    status_class = "status-warning" if result["has_missing_clauses"] else "status-ok"
                    status_text = "Missing Clauses Found" if result["has_missing_clauses"] else "No Missing Clauses"
                    st.markdown(f'<div class="status-indicator {status_class}">{status_text}</div>', unsafe_allow_html=True)
                
                # Analysis
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown(result["analysis"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Discrepancies with severity and solutions
                if "discrepancies" in result and result["discrepancies"]:
                    st.subheader("Detailed Discrepancies")
                    
                    # Separate discrepancies by action needed
                    action_needed = []
                    acceptable = []
                    
                    for disc in result["discrepancies"]:
                        if disc.get("action_needed", True):
                            action_needed.append(disc)
                        else:
                            acceptable.append(disc)
                    
                    # Display discrepancies that need action first
                    if action_needed:
                        st.markdown("### Discrepancies Requiring Action")
                        for disc in action_needed:
                            severity_class = f"severity-{disc['severity'].lower()}" if "severity" in disc else "severity-minor"
                            
                            with st.expander(f"{disc.get('description', 'Discrepancy')} [{disc.get('severity', 'MINOR')}] - ACTION NEEDED"):
                                st.markdown(f"**Description:** {disc.get('description', 'No description available')}")
                                st.markdown(f"**Severity:** <span class='{severity_class}'>{disc.get('severity', 'MINOR')}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Action Required:** <span class='action-needed'>YES</span>", unsafe_allow_html=True)
                                st.markdown(f"**Rationale:** {disc.get('rationale', 'No rationale provided')}")
                                st.markdown(f"**Proposed Solution:** {disc.get('solution', 'No solution provided')}")
                    
                    # Display acceptable discrepancies
                    if acceptable:
                        st.markdown("### Acceptable Discrepancies")
                        for disc in acceptable:
                            severity_class = f"severity-{disc['severity'].lower()}" if "severity" in disc else "severity-minor"
                            
                            with st.expander(f"{disc.get('description', 'Discrepancy')} [{disc.get('severity', 'MINOR')}] - ACCEPTABLE"):
                                st.markdown(f"**Description:** {disc.get('description', 'No description available')}")
                                st.markdown(f"**Severity:** <span class='{severity_class}'>{disc.get('severity', 'MINOR')}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Action Required:** <span class='acceptable'>NO</span>", unsafe_allow_html=True)
                                st.markdown(f"**Rationale:** {disc.get('rationale', 'No rationale provided')}")
                
                # Source sections
                with st.expander("View Source Sections"):
                    st.subheader("Master Document")
                    st.markdown(f"```\n{result['master_sections']}\n```")
                    
                    st.subheader("Client Document")
                    st.markdown(f"```\n{result['client_sections']}\n```")
        
        # Export options
        st.markdown("---")
        st.subheader("Export Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Generate report content
            report_content = f"# T&C Sentinel - Comparison Report\n\n"
            report_content += f"**Master Document:** {st.session_state.master_filename}\n"
            report_content += f"**Client Document:** {st.session_state.client_filename}\n\n"
            report_content += f"## Executive Summary\n\n{st.session_state.comparison_results['overall_summary']}\n\n"
            
            for category in categories:
                result = st.session_state.comparison_results[category]
                report_content += f"## {category}\n\n{result['analysis']}\n\n"
                
                # Add discrepancies details
                if "discrepancies" in result and result["discrepancies"]:
                    # Separate discrepancies by action needed
                    action_needed = []
                    acceptable = []
                    
                    for disc in result["discrepancies"]:
                        if disc.get("action_needed", True):
                            action_needed.append(disc)
                        else:
                            acceptable.append(disc)
                    
                    # Add discrepancies that need action
                    if action_needed:
                        report_content += "### Discrepancies Requiring Action\n\n"
                        for disc in action_needed:
                            report_content += f"**{disc.get('description', 'Discrepancy')} [{disc.get('severity', 'MINOR')}] - ACTION NEEDED**\n\n"
                            report_content += f"- **Severity:** {disc.get('severity', 'MINOR')}\n"
                            report_content += f"- **Rationale:** {disc.get('rationale', 'No rationale provided')}\n"
                            report_content += f"- **Proposed Solution:** {disc.get('solution', 'No solution provided')}\n\n"
                    
                    # Add acceptable discrepancies
                    if acceptable:
                        report_content += "### Acceptable Discrepancies\n\n"
                        for disc in acceptable:
                            report_content += f"**{disc.get('description', 'Discrepancy')} [{disc.get('severity', 'MINOR')}] - ACCEPTABLE**\n\n"
                            report_content += f"- **Severity:** {disc.get('severity', 'MINOR')}\n"
                            report_content += f"- **Rationale:** {disc.get('rationale', 'No rationale provided')}\n\n"
            
            # Provide download button for markdown report
            st.download_button(
                label="Download Markdown Report",
                data=report_content,
                file_name="comparison_report.md",
                mime="text/markdown"
            )
        
        with export_col2:
            # Generate annotated PDF if client path is available
            if hasattr(st.session_state, 'client_path') and os.path.exists(st.session_state.client_path):
                # Collect all discrepancies
                all_discrepancies = []
                for category in categories:
                    if "discrepancies" in st.session_state.comparison_results[category]:
                        all_discrepancies.extend(st.session_state.comparison_results[category]["discrepancies"])
                
                if st.button("Generate Annotated PDF"):
                    with st.spinner("Generating annotated PDF..."):
                        # Generate annotated PDF
                        output_path, pdf_bytes = comparison_engine.generate_annotated_pdf(
                            pdf_path=st.session_state.client_path,
                            discrepancies=all_discrepancies
                        )
                        
                        # Create download link
                        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="annotated_{st.session_state.client_filename}">Download Annotated PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        st.success("Annotated PDF generated!")

if __name__ == "__main__":
    main()