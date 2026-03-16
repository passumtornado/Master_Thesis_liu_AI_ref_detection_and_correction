import asyncio

from agents.validation_agent import DBLPValidationAgent
from utils.google_scholar import compute_scholar_similarity, extract_scholar_hits


SAMPLE_SCHOLAR_HTML = """
<html>
  <body>
    <div class="gs_r gs_or gs_scl">
      <div class="gs_ri">
        <h3 class="gs_rt">
          <a href="https://example.org/paper">Advanced SMT techniques for weighted model integration</a>
        </h3>
        <div class="gs_a">Paolo Morettin, Andrea Passerini, Roberto Sebastiani - Artificial Intelligence, 2019</div>
        <div class="gs_rs">Weighted model integration paper.</div>
        <div class="gs_fl">links</div>
      </div>
    </div>
    <div class="gs_r gs_or gs_scl">
      <div class="gs_ri">
        <h3 class="gs_rt">
          <a href="https://example.org/other">Completely unrelated title</a>
        </h3>
        <div class="gs_a">Someone Else - Workshop Notes, 2015</div>
        <div class="gs_rs">Other paper.</div>
        <div class="gs_fl">links</div>
      </div>
    </div>
  </body>
</html>
"""


def test_extract_scholar_hits_prefers_best_match():
    hits = extract_scholar_hits(
        SAMPLE_SCHOLAR_HTML,
        query_title="Advanced SMT techniques for weighted model integration",
        query_author="Paolo Morettin and Andrea Passerini and Roberto Sebastiani",
        query_year="2019",
    )

    assert len(hits) == 2
    assert hits[0]["title"] == "Advanced SMT techniques for weighted model integration"
    assert hits[0]["year"] == "2019"
    assert hits[0]["similarity_score"] >= 0.85
    assert hits[0]["similarity_score"] > hits[1]["similarity_score"]


def test_compute_scholar_similarity_rewards_metadata_alignment():
    score = compute_scholar_similarity(
        query_title="Attention Is All You Need",
        candidate_title="Attention Is All You Need",
        query_author="Ashish Vaswani and Noam Shazeer",
        candidate_authors="A Vaswani, N Shazeer",
        query_year="2017",
        candidate_year="2017",
    )

    assert score >= 0.85


def test_validation_agent_uses_scholar_fallback(monkeypatch):
    async def fake_title_search(self, tools, title, year):
        return []

    async def fake_author_search(self, tools, author, title):
        return None

    async def fake_explain(self, entry, hit, match_source, status, confidence, issues):
        return f"{match_source}:{status}:{confidence}"

    monkeypatch.setattr(DBLPValidationAgent, "_title_search", fake_title_search)
    monkeypatch.setattr(DBLPValidationAgent, "_author_search", fake_author_search)
    monkeypatch.setattr(DBLPValidationAgent, "_explain", fake_explain)
    monkeypatch.setattr(
        "agents.validation_agent.search_google_scholar",
        lambda **kwargs: [
            {
                "title": "Advanced SMT techniques for weighted model integration",
                "authors": "Paolo Morettin, Andrea Passerini, Roberto Sebastiani",
                "year": "2019",
                "venue": "Artificial Intelligence",
                "url": "https://example.org/paper",
                "snippet": "Weighted model integration paper.",
                "similarity_score": 0.96,
            }
        ],
    )

    agent = DBLPValidationAgent("server/mcp.json")
    entry = {
        "id": "Morettin2019",
        "type": "article",
        "title": "Advanced SMT techniques for weighted model integration",
        "author": "Paolo Morettin and Andrea Passerini and Roberto Sebastiani",
        "year": "2019",
        "journal": "Artificial Intelligence",
    }

    result = asyncio.run(agent._validate_entry({}, entry))

    assert result["status"] == "valid"
    assert result["match_source"] == "google_scholar"
    assert result["dblp_match"] == {}
    assert result["scholar_match"]["title"] == entry["title"]
    assert result["confidence"] > 0.75