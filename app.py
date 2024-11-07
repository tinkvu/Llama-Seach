import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
from typing import List, Dict
from groq import Groq

class SearchValidationSystem:
    def __init__(self, groq_api_key: str):
        """Initialize the system with Groq API key."""
        self.groq_client = Groq(api_key=groq_api_key)
        self.search_results_cache = {}

    # [Previous SearchValidationSystem methods remain the same]
    def fetch_duckduckgo_lite_results(self, query: str, num_results: int = 5) -> List[Dict]:
        """Fetch search results from DuckDuckGo Lite."""
        url = "https://lite.duckduckgo.com/lite"
        results = []
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            with st.spinner('Fetching search results...'):
                response = requests.post(url, headers=headers, data={'q': query})
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', limit=num_results)
                
                progress_bar = st.progress(0)
                for i, link in enumerate(links):
                    title = link.get_text(strip=True)
                    result_url = link.get('href')
                    
                    if result_url and title:
                        content = self._fetch_page_content(result_url)
                        results.append({
                            'url': result_url,
                            'title': title,
                            'description': content[:150] + "...",
                            'timestamp': datetime.now().isoformat()
                        })
                    progress_bar.progress((i + 1) / len(links))
                    time.sleep(1)

        except Exception as e:
            st.error(f"Error in DuckDuckGo Lite search: {e}")

        return results

    def _fetch_page_content(self, url: str) -> str:
        """Fetch and parse content from a webpage."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                return content[:5000]
            return ""

        except Exception as e:
            st.warning(f"Error fetching content from {url}: {e}")
            return ""

    def validate_with_llama(self, query: str, search_results: List[Dict]) -> Dict:
        """Validate search results using LLaMA through Groq."""
        if not search_results:
            return {"error": "No valid search results to analyze"}

        context = "\n".join([
            f"Source {i+1} ({result['url']}):\nTitle: {result['title']}\nDescription: {result['description']}\nContent: {result['description'][:200]}..."
            for i, result in enumerate(search_results)
        ])

        prompt = f"""
        Query: {query}
        
        Context from multiple sources:
        {context}
        
        Please analyze the above information and provide:
        1. A summary of the key findings
        2. Validation of the information across sources
        3. Any inconsistencies or contradictions
        4. List of reliable reference links
        
        Format your response as JSON with the following structure:
        {{
            "summary": "key findings",
            "validation": "analysis of information validity",
            "inconsistencies": ["list of any contradictions"],
            "references": ["list of verified urls"]
        }}
        """

        try:
            with st.spinner('Validating information with LLaMA...'):
                response = self.groq_client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    max_tokens=2048
                )

                if not response.choices or not response.choices[0].message.content:
                    st.error("Empty response received from Groq API.")
                    return {
                        "summary": "Error in validation process",
                        "validation": "Empty response from Groq API",
                        "inconsistencies": [],
                        "references": []
                    }

                try:
                    return json.loads(response.choices[0].message.content)
                except json.JSONDecodeError:
                    st.error("Error decoding the JSON response from Groq.")
                    return {
                        "summary": "Error in validation process",
                        "validation": "Invalid JSON response from Groq",
                        "inconsistencies": [],
                        "references": []
                    }

        except Exception as e:
            st.error(f"Error in LLaMA validation: {e}")
            return {
                "summary": "Error in validation process",
                "validation": str(e),
                "inconsistencies": [],
                "references": []
            }

    def search_and_validate(self, query: str) -> Dict:
        """Main method to perform search and validation."""
        if query in self.search_results_cache:
            st.info("Using cached results...")
            return self.search_results_cache[query]

        search_results = self.fetch_duckduckgo_lite_results(query)
        if not search_results:
            return {"error": "No search results found"}

        validation_results = self.validate_with_llama(query, search_results)
        
        final_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "search_results": search_results,
            "validation": validation_results
        }

        self.search_results_cache[query] = final_results
        return final_results

def main():
    st.set_page_config(
        page_title="Search Validation System",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Search Validation System")
    st.markdown("""
    This app fetches search results and validates them using LLaMA AI model.
    Enter your query below to get started!
    """)

    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
        num_results = st.slider("Number of search results:", min_value=3, max_value=10, value=5)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app combines DuckDuckGo search with LLaMA validation through Groq's API
        to provide verified information with cross-referenced sources.
        """)

    # Main interface
    query = st.text_input("Enter your search query:", key="search_query")

    if query and groq_api_key:
        system = SearchValidationSystem(groq_api_key)
        
        try:
            results = system.search_and_validate(query)
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                # Display results in tabs
                tab1, tab2 = st.tabs(["Search Results", "Validation"])
                
                with tab1:
                    st.subheader("Search Results")
                    for i, result in enumerate(results['search_results'], 1):
                        with st.expander(f"{i}. {result['title']}", expanded=True):
                            st.write(f"**URL:** {result['url']}")
                            st.write(f"**Description:** {result['description']}")
                            st.write(f"**Timestamp:** {result['timestamp']}")
                
                with tab2:
                    st.subheader("Validation Results")
                    validation = results['validation']
                    if 'error' in validation:
                        st.error(f"Validation Error: {validation['error']}")
                    else:
                        st.markdown("### Summary")
                        st.write(validation['summary'])
                        
                        st.markdown("### Validation Details")
                        st.write(validation['validation'])
                        
                        if validation['inconsistencies']:
                            st.markdown("### Inconsistencies Found")
                            for inconsistency in validation['inconsistencies']:
                                st.warning(f"- {inconsistency}")
                        
                        if validation['references']:
                            st.markdown("### References")
                            for ref in validation['references']:
                                st.markdown(f"- {ref}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    elif query:
        st.warning("Please enter your Groq API key in the sidebar to proceed.")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, DuckDuckGo, and LLaMA")

if __name__ == "__main__":
    main()
