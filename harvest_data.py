import trafilatura
import json
import os
import time
import re
import logging
import random
from urllib.parse import urljoin, urlparse
from typing import List, Set, Dict, Optional
from datasketch import MinHash, MinHashLSH
from textstat import textstat
from trafilatura.settings import use_config
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("harvester.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "architect_training_data.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
MAX_PAGES = 5000  # Increased as per recommendation
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential backoff: 2s, 4s, 8s

INSTRUCTION_TEMPLATES = [
    "As a Senior Cloud Security Architect, explain the concepts regarding {topic}.",
    "What are the security best practices for {topic} based on this documentation?",
    "Design a secure architecture incorporating {topic} using the following reference.",
    "Analyze the compliance implications of {topic}.",
    "What are the risks and mitigations for {topic}?",
    "How would you implement {topic} following Zero Trust principles?",
]

START_URLS = [
    "https://docs.aws.amazon.com/prescriptive-guidance/latest/security-reference-architecture/welcome.html",
    "https://learn.microsoft.com/en-us/azure/well-architected/security/",
    "https://cloud.google.com/architecture/framework/security",
    "https://kubernetes.io/docs/concepts/security/",
    "https://attack.mitre.org/matrices/enterprise/cloud/",
    "https://www.cisecurity.org/controls/cis-controls-list",
]

ALLOWED_DOMAINS = [
    "docs.aws.amazon.com", "learn.microsoft.com", "cloud.google.com",
    "kubernetes.io", "attack.mitre.org", "www.cisecurity.org"
]

class DataHarvester:
    def __init__(self):
        self.lsh = MinHashLSH(threshold=0.85, num_perm=128)
        self.visited: Set[str] = set()
        self.queue: List[str] = list(START_URLS)
        self.collected_count = 0
        
        # Configure Trafilatura with User Agent
        self.traf_config = use_config()
        self.traf_config.set("DEFAULT", "USER_AGENT", "CloudSecArchAI-Harvester/1.0 (Research)")
        
        self._load_checkpoint()

    def _get_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    def _is_duplicate(self, text: str, doc_id: str) -> bool:
        m = self._get_minhash(text)
        if len(self.lsh.query(m)) > 0:
            return True
        self.lsh.insert(doc_id, m)
        return False

    def _save_checkpoint(self):
        """Saves current state to allow resuming."""
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "visited": list(self.visited),
                "queue": self.queue,
                "collected_count": self.collected_count
            }, f)
        logger.info("Checkpoint saved.")

    def _load_checkpoint(self):
        """Loads state if checkpoint exists."""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    data = json.load(f)
                    self.visited = set(data["visited"])
                    self.queue = data["queue"]
                    self.collected_count = data["collected_count"]
                logger.info(f"Resumed from checkpoint. Collected: {self.collected_count}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            while self.queue and self.collected_count < MAX_PAGES:
                url = self.queue.pop(0)
                if url in self.visited: continue
                
                domain = urlparse(url).netloc
                if domain not in ALLOWED_DOMAINS: continue

                self.visited.add(url)
                logger.info(f"[{self.collected_count}] Processing: {url}")

                # Retry logic with exponential backoff for network errors
                downloaded = None
                for attempt in range(MAX_RETRIES):
                    try:
                        downloaded = trafilatura.fetch_url(url, config=self.traf_config)
                        if not downloaded:
                            logger.warning(f"Empty response from {url}")
                        break  # Success, exit retry loop
                    except (RequestException, Timeout, ConnectionError) as e:
                        wait_time = BACKOFF_FACTOR ** attempt
                        logger.warning(f"Network error (attempt {attempt+1}/{MAX_RETRIES}) for {url}: {e}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed after {MAX_RETRIES} attempts: {url}")
                            downloaded = None
                    except ValueError as e:
                        logger.error(f"Invalid URL or parsing error for {url}: {e}")
                        downloaded = None
                        break
                    except Exception as e:
                        logger.critical(f"Unexpected error fetching {url}: {e}", exc_info=True)
                        downloaded = None
                        break

                if not downloaded:
                    continue

                try:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, no_fallback=True)
                    if not text:
                        logger.debug(f"No extractable text from {url}")
                        continue

                    # Quality Gates
                    if len(text) < 500:
                        logger.debug(f"Text too short ({len(text)} chars) from {url}")
                        continue

                    if textstat.flesch_kincaid_grade(text) < 8:
                        logger.debug(f"Text grade level too low from {url}")
                        continue

                    if self._is_duplicate(text, url):
                        logger.debug(f"Duplicate content found: {url}")
                        continue

                    # Chunking & Instruction Selection
                    chunks = text.split('\n\n')
                    buffer = ""

                    for chunk in chunks:
                        if len(buffer) + len(chunk) < 2000:
                            buffer += chunk + "\n\n"
                        else:
                            if len(buffer) > 200:
                                # Select a random diverse instruction
                                template = random.choice(INSTRUCTION_TEMPLATES)
                                instruction = template.format(topic=domain)

                                entry = {
                                    "instruction": instruction,
                                    "input": f"Source: {url}",
                                    "output": buffer.strip()
                                }
                                f.write(json.dumps(entry) + "\n")
                                self.collected_count += 1
                            buffer = chunk

                    # Link Discovery
                    for link in re.findall(r'href=[\'"]?([^\'" >]+)', downloaded):
                        full_link = urljoin(url, link)
                        if urlparse(full_link).netloc in ALLOWED_DOMAINS and full_link not in self.visited:
                            self.queue.append(full_link)

                    # Periodically save checkpoint
                    if self.collected_count % 50 == 0:
                        self._save_checkpoint()

                    time.sleep(1.0)  # Polite rate limiting

                except Exception as e:
                    logger.error(f"Error processing content from {url}: {e}", exc_info=True)
                    continue

        logger.info(f"Harvest Complete. Saved {self.collected_count} samples.")

if __name__ == "__main__":
    harvester = DataHarvester()
    harvester.run()