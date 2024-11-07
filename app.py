import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
from typing import List, Dict
from groq import Groq

class SearchValidationSystem:
    def __init__(self):
        """Initialize the system with Groq API key directly."""
        self.groq_client = Groq(api_key="YOUR_GROQ_API_KEY")  # Replace with your Groq API key
        self.search_results_cache = {}

    def fetch_duckduckgo_lite_results(self, query: str, num_results: int = 5) -> List[Dict]:
        """Fetch search results from DuckDuckGo Lite."""
        url = "https://lite.duckduckgo.com/lite"
        results = []
        
        # Setting headers to simulate a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            # Submit the search form
            response = requests.post(url, headers=headers, data={'q': query})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Locate search result elements
            links = soup.find_all('a', limit=num_results)
            
            for link in links:
                title = link.get_text(strip=True)
                result_url = link.get('href')
                
                # Only add valid links
                if result_url and title:
                    # Fetch and parse content for each link
                    content = self._fetch_page_content(result_url)
                    results.append({
                        'url': result_url,
                        'title': title,
                        'description': content[:150] + "...",  # Shorten description for display
                        'timestamp': datetime.now().isoformat()
                    })

                # Add a delay to avoid triggering anti-bot measures
                time.sleep(1)

        except Exception as e:
            print(f"Error in DuckDuckGo Lite search: {e}")

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
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):  # Clean unnecessary content
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                return content[:5000]
            return ""

        except Exception as e:
            print(f"Error fetching page content from {url}: {e}")
            return ""

    def validate_with_llama(self, query: str, search_results: List[Dict]) -> Dict:
        """Validate search results using LLaMA through Groq."""
        # Check if the search results contain content
        if not search_results:
            return {"error": "No valid search results to analyze"}

        # Prepare context from search results
        context = "\n".join([f"Source {i+1} ({result['url']}):\nTitle: {result['title']}\nDescription: {result['description']}\nContent: {result['description'][:200]}..."
                            for i, result in enumerate(search_results)])

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
            content = response.choices[0].message.content
            return json.loads(content) if content else {"error": "Empty response from LLaMA"}

        except Exception as e:
            print(f"Error in LLaMA validation: {e}")
            return {
                "summary": "Error in validation process",
                "validation": str(e),
                "inconsistencies": [],
                "references": []
            }

    def search_and_validate(self, query: str) -> Dict:
        """Main method to perform search and validation."""
        # Check cache first
        if query in self.search_results_cache:
            print("Using cached results...")
            return self.search_results_cache[query]

        # Fetch new results
        search_results = self.fetch_duckduckgo_lite_results(query)
        if not search_results:
            return {"error": "No search results found"}

        # Validate with LLaMA
        validation_results = self.validate_with_llama(query, search_results)
        
        # Combine results
        final_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "search_results": search_results,
            "validation": validation_results
        }

        # Cache results
        self.search_results_cache[query] = final_results
        return final_results


def main():
    # Initialize Streamlit app UI
    system = SearchValidationSystem()

    st.title("Search and Validation System")
    query = st.text_input("Enter search query:", value="US Election 2024")

    if st.button("Search and Validate"):
        if not query:
            st.error("Please enter a search query.")
        else:
            results = system.search_and_validate(query)

            if "error" in results:
                st.error(results["error"])
            else:
                st.subheader(f"Results for: {results['query']}")
                st.write(f"Timestamp: {results['timestamp']}")

                st.subheader("Search Results:")
                for i, result in enumerate(results['search_results'], 1):
                    st.write(f"{i}. **{result['title']}**")
                    st.write(f"   URL: {result['url']}")
                    st.write(f"   Description: {result['description']}")
                    st.write(f"   Timestamp: {result['timestamp']}\n")

                st.subheader("Validation Results:")
                validation = results['validation']
                st.write("Summary:")
                st.write(f"  {validation['summary']}")
                st.write("Validation Details:")
                st.write(f"  {validation['validation']}")
                
                if validation['inconsistencies']:
                    st.write("Inconsistencies Found:")
                    for inconsistency in validation['inconsistencies']:
                        st.write(f"  - {inconsistency}")

                if validation['references']:
                    st.write("References:")
                    for ref in validation['references']:
                        st.write(f"  - {ref}")


if __name__ == "__main__":
    main()

