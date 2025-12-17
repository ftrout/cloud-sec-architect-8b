import trafilatura
import json
import os
import time
import re
from urllib.parse import urljoin, urlparse
from datasketch import MinHash, MinHashLSH
from textstat import textstat

# --- CONFIGURATION ---
OUTPUT_DIR = "./data_architect"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "architect_training_data.jsonl")
MAX_PAGES = 3000

# The "Pro" List: Cloud + Identity + IaC + Standards
START_URLS = [
    # --- The Big 3 Clouds (Architecture Centers) ---
    "https://docs.aws.amazon.com/prescriptive-guidance/latest/security-reference-architecture/welcome.html",
    "https://learn.microsoft.com/en-us/azure/well-architected/security/",
    "https://cloud.google.com/architecture/framework/security",
    
    # --- Container & Runtime Security ---
    "https://kubernetes.io/docs/concepts/security/",
    "https://docs.docker.com/security/",
    "https://falco.org/docs/",
    
    # --- Data Security ---
    "https://docs.databricks.com/en/security/index.html",
    "https://docs.snowflake.com/en/user-guide/security-overview",
    
    # --- Identity (The New Perimeter) ---
    "https://auth0.com/docs/secure",
    "https://developer.okta.com/docs/concepts/",

    # --- Frameworks & Threat Models ---
    "https://attack.mitre.org/matrices/enterprise/cloud/",
    "https://owasp.org/www-project-top-ten/",
    "https://www.cisecurity.org/controls/cis-controls-list",
    
    # --- Infrastructure as Code (IaC) ---
    "https://developer.hashicorp.com/terraform/tutorials/security",
    "https://www.openpolicyagent.org/docs/latest/security/"
]

ALLOWED_DOMAINS = [
    "docs.aws.amazon.com", "learn.microsoft.com", "cloud.google.com",
    "kubernetes.io", "docs.docker.com", "falco.org",
    "docs.databricks.com", "docs.snowflake.com",
    "auth0.com", "developer.okta.com",
    "attack.mitre.org", "owasp.org", "cisecurity.org",
    "developer.hashicorp.com", "www.openpolicyagent.org"
]

# --- DEDUPLICATION SETUP ---
lsh = MinHashLSH(threshold=0.85, num_perm=128)

def get_minhash(text):
    m = MinHash(num_perm=128)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def is_duplicate(text, doc_id):
    m = get_minhash(text)
    if len(lsh.query(m)) > 0: return True
    lsh.insert(doc_id, m)
    return False

# --- CRAWLER LOGIC ---
def crawl_and_process():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    visited = set()
    queue = list(START_URLS)
    collected_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        while queue and collected_count < MAX_PAGES:
            url = queue.pop(0)
            if url in visited: continue
            
            domain = urlparse(url).netloc
            if domain not in ALLOWED_DOMAINS: continue

            visited.add(url)
            print(f"[{collected_count}] Processing: {url}")
            
            try:
                # 1. FETCH & EXTRACT
                downloaded = trafilatura.fetch_url(url)
                if not downloaded: continue
                
                text_content = trafilatura.extract(
                    downloaded, include_comments=False, include_tables=True, no_fallback=True
                )
                if not text_content: continue

                # 2. QUALITY GATES
                # Must be substantial content (>500 chars) and technical (> Grade 8 reading level)
                if len(text_content) < 500: continue 
                if textstat.flesch_kincaid_grade(text_content) < 8: continue
                if is_duplicate(text_content, url): continue

                # 3. CHUNKING & FORMATTING
                chunks = text_content.split('\n\n')
                buffer = ""
                
                for chunk in chunks:
                    if len(buffer) + len(chunk) < 2000:
                        buffer += chunk + "\n\n"
                    else:
                        if len(buffer) > 200:
                            # Context Injection for RAG-like behavior
                            entry = {
                                "instruction": f"Explain the security architecture concepts regarding {domain}.",
                                "input": f"Source: {url}",
                                "output": buffer.strip()
                            }
                            f.write(json.dumps(entry) + "\n")
                            collected_count += 1
                        buffer = chunk

                # 4. LINK DISCOVERY
                for link in re.findall(r'href=[\'"]?([^\'" >]+)', downloaded):
                    full_link = urljoin(url, link)
                    if urlparse(full_link).netloc in ALLOWED_DOMAINS:
                        if full_link not in visited:
                            queue.append(full_link)

                time.sleep(0.3)

            except Exception as e:
                print(f"Skipping {url}: {e}")

    print(f"Harvest Complete. Saved {collected_count} samples.")

if __name__ == "__main__":
    crawl_and_process()