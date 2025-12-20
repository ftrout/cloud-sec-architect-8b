import pytest
from datasketch import MinHash
from harvester import DataHarvester  # Assuming harvester logic is imported

def test_deduplication():
    harvester = DataHarvester()
    
    text1 = "This is a security document about AWS IAM."
    text2 = "This is a security document about AWS IAM." # Exact duplicate
    text3 = "This is a completely different document about Azure."
    
    assert harvester._is_duplicate(text1, "url1") == False
    assert harvester._is_duplicate(text2, "url2") == True
    assert harvester._is_duplicate(text3, "url3") == False

def test_instruction_formatting():
    # Test that templates are formatted correctly
    domain = "aws.amazon.com"
    templates = [
        "Explain {topic}",
        "Security for {topic}"
    ]
    formatted = [t.format(topic=domain) for t in templates]
    assert "Explain aws.amazon.com" in formatted