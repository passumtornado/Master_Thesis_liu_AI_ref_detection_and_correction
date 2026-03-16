#!/usr/bin/env python3
"""Test the DBLP text parser."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.validation_agent import _parse_dblp_text_response

# Test with real output from earlier
text = """Found 1 publications with similar titles:

1. Advanced SMT techniques for weighted model integration. [Similarity: 0.99]
   Authors: Paolo Morettin, Andrea Passerini, Roberto Sebastiani
   Venue: Artif. Intell. (2019)
"""

results = _parse_dblp_text_response(text)
print(f"Parsed {len(results)} results:\n")
for i, r in enumerate(results, 1):
    print(f"Result {i}:")
    for key, val in r.items():
        print(f"  {key}: {val}")
    print()

# Test with multiple results
text_multi = """Found 3 publications with similar titles:

1. Advanced SMT techniques. [Similarity: 0.99]
   Authors: Author A, Author B
   Venue: Journal X (2019)

2. Another paper. [Similarity: 0.75]
   Authors: Author C
   Venue: Conference Y (2020)

3. Third paper. [Similarity: 0.50]
   Authors: Author D, Author E, Author F
   Venue: Workshop Z (2021)
"""

print("\n\nTesting multiple results:")
results_multi = _parse_dblp_text_response(text_multi)
print(f"Parsed {len(results_multi)} results:\n")
for i, r in enumerate(results_multi, 1):
    print(f"Result {i}: {r['title'][:40]} (sim={r['similarity_score']})")
