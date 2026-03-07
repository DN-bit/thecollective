# The Collective - RAG Module
# Embeds and retrieves relevant 2 Satoshis posts for MacroNode context injection
# Uses Voyage AI for embeddings (voyage-finance-2 model - optimized for financial text)

import asyncio
import json
import math
import os
from typing import Optional

import asyncpg
import httpx

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://")

VOYAGE_BASE = "https://api.voyageai.com/v1"
EMBEDDING_MODEL = "voyage-finance-2"  # Finance-optimized model, perfect for crypto/investment text
EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

async def get_conn():
    return await asyncpg.connect(DATABASE_URL)


async def init_embedding_column():
    """Add embedding column to arca_posts if not exists."""
    conn = await get_conn()
    await conn.execute("""
        ALTER TABLE arca_posts 
        ADD COLUMN IF NOT EXISTS embedding TEXT
    """)
    await conn.close()


# ---------------------------------------------------------------------------
# Voyage AI Embeddings
# ---------------------------------------------------------------------------

async def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    Get embeddings from Voyage AI.
    input_type: "document" for posts being indexed, "query" for search queries
    """
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Voyage has a 128 input limit per request, batch if needed
    all_embeddings = []
    batch_size = 64

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate to ~4000 chars to stay within token limits
            batch = [t[:4000] for t in batch]

            resp = await client.post(
                f"{VOYAGE_BASE}/embeddings",
                headers=headers,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                    "input_type": input_type
                }
            )
            if resp.status_code != 200:
                raise Exception(f"Voyage API error: {resp.status_code} {resp.text}")

            data = resp.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
            print(f"[RAG] Embedded batch {i//batch_size + 1} ({len(batch)} texts)")

    return all_embeddings


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

async def index_posts():
    """
    Embed all unembedded posts and store embeddings in DB.
    Safe to call multiple times - only processes posts without embeddings.
    """
    await init_embedding_column()

    conn = await get_conn()
    posts = await conn.fetch("""
        SELECT id, title, content 
        FROM arca_posts 
        WHERE embedding IS NULL
        ORDER BY id
    """)
    await conn.close()

    if not posts:
        print("[RAG] All posts already indexed")
        return 0

    print(f"[RAG] Indexing {len(posts)} posts...")

    # Prepare texts: title + first 3000 chars of content for embedding
    texts = []
    for p in posts:
        text = f"{p['title']}\n\n{p['content'][:3000]}" if p['title'] else p['content'][:3000]
        texts.append(text)

    embeddings = await embed_texts(texts, input_type="document")

    # Store embeddings
    conn = await get_conn()
    for post, embedding in zip(posts, embeddings):
        await conn.execute("""
            UPDATE arca_posts SET embedding = $1 WHERE id = $2
        """, json.dumps(embedding), post['id'])
    await conn.close()

    print(f"[RAG] Indexed {len(posts)} posts successfully")
    return len(posts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve_relevant_posts(query: str, top_k: int = 3) -> list[dict]:
    """
    Find the most semantically relevant 2 Satoshis posts for a given query.
    Returns list of {title, url, published_date, excerpt, similarity} dicts.
    """
    if not VOYAGE_API_KEY:
        print("[RAG] WARNING: VOYAGE_API_KEY not set, skipping retrieval")
        return []

    # Embed the query
    query_embeddings = await embed_texts([query], input_type="query")
    query_embedding = query_embeddings[0]

    # Fetch all embedded posts
    conn = await get_conn()
    posts = await conn.fetch("""
        SELECT id, title, url, published_date, content, embedding
        FROM arca_posts
        WHERE embedding IS NOT NULL
    """)
    await conn.close()

    if not posts:
        print("[RAG] No indexed posts found")
        return []

    # Score by cosine similarity
    scored = []
    for post in posts:
        try:
            post_embedding = json.loads(post['embedding'])
            score = cosine_similarity(query_embedding, post_embedding)
            scored.append({
                "title": post['title'],
                "url": post['url'],
                "published_date": post['published_date'],
                "excerpt": post['content'][:800],
                "similarity": score
            })
        except Exception as e:
            print(f"[RAG] WARNING: Failed to score post {post['id']}: {e}")

    # Sort by similarity, return top_k
    scored.sort(key=lambda x: x['similarity'], reverse=True)
    top = scored[:top_k]

    print(f"[RAG] Retrieved {len(top)} relevant posts (top score: {top[0]['similarity']:.3f})")
    return top


# ---------------------------------------------------------------------------
# Context Formatter
# ---------------------------------------------------------------------------

def format_rag_context(posts: list[dict]) -> str:
    """
    Format retrieved posts as context for MacroNode prompt injection.
    """
    if not posts:
        return ""

    lines = ["═══ RELEVANT ARCA RESEARCH (2 Satoshis) ═══",
             "The following past Arca analyses are semantically relevant to this event.",
             "Use them to ground your analysis in Arca's established frameworks and voice.\n"]

    for i, post in enumerate(posts, 1):
        date_str = post['published_date'][:10] if post['published_date'] else "unknown date"
        lines.append(f"[{i}] {post['title']} ({date_str})")
        lines.append(f"URL: {post['url']}")
        lines.append(f"Relevance score: {post['similarity']:.2f}")
        lines.append(f"Excerpt:\n{post['excerpt']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main (for standalone indexing run)
# ---------------------------------------------------------------------------

async def main():
    print("[RAG] Starting post indexing...")
    count = await index_posts()
    print(f"[RAG] Done. Indexed {count} new posts.")

    # Test retrieval
    print("\n[RAG] Testing retrieval with sample query...")
    results = await retrieve_relevant_posts("Federal Reserve interest rate crypto market impact")
    for r in results:
        print(f"  - {r['title']} (score: {r['similarity']:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
