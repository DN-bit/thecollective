# The Collective - Portfolio Impact System
# Fetches top 100 tokens from CoinGecko, stores in DB, runs impact analysis

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

import httpx
import asyncpg

logging.basicConfig(level=logging.INFO, format='[Portfolio] %(asctime)s %(message)s')
log = logging.getLogger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

# Sector inference from CoinGecko categories/tags
SECTOR_MAP = {
    "layer-1": ["bitcoin", "ethereum", "solana", "avalanche", "cardano", "polkadot", "near", "cosmos", "tron", "aptos", "sui", "algorand", "hedera"],
    "layer-2": ["polygon", "arbitrum", "optimism", "base", "starknet", "zksync", "mantle", "scroll"],
    "defi": ["uniswap", "aave", "maker", "compound", "curve", "synthetix", "yearn", "balancer", "1inch", "pancakeswap", "jupiter", "raydium", "hyperliquid"],
    "exchange": ["binancecoin", "okb", "crypto-com-chain", "kucoin-shares", "gate", "bitget-token"],
    "stablecoin": ["tether", "usd-coin", "dai", "first-digital-usd", "ethena-usde", "paypal-usd"],
    "ai": ["fetch-ai", "singularitynet", "ocean-protocol", "render-token", "bittensor", "akash-network", "artificial-superintelligence-alliance"],
    "gaming": ["axie-infinity", "the-sandbox", "decentraland", "gala", "immutable-x", "beam"],
    "infrastructure": ["chainlink", "filecoin", "the-graph", "helium", "arweave", "storj", "livepeer"],
    "meme": ["dogecoin", "shiba-inu", "pepe", "bonk", "dogwifcoin", "floki"],
}

def infer_sector(coin_id: str, name: str) -> str:
    cid = coin_id.lower()
    for sector, ids in SECTOR_MAP.items():
        if cid in ids:
            return sector
    return "other"


async def init_assets_table(conn):
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            id SERIAL PRIMARY KEY,
            coin_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT,
            market_cap_rank INTEGER,
            market_cap_usd FLOAT,
            current_price_usd FLOAT,
            price_change_24h FLOAT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS portfolio_impacts (
            id SERIAL PRIMARY KEY,
            event_id TEXT NOT NULL,
            coin_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            impact_direction TEXT,
            impact_severity TEXT,
            rationale TEXT,
            confidence FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(event_id, coin_id)
        );
    """)
    log.info("Assets and portfolio_impacts tables ready")


async def fetch_top_100() -> list:
    async with httpx.AsyncClient() as client:
        resp = await client.get(COINGECKO_URL, params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False
        }, timeout=15)
        return resp.json()


async def sync_assets(conn):
    log.info("Fetching top 100 tokens from CoinGecko...")
    coins = await fetch_top_100()
    count = 0
    for coin in coins:
        sector = infer_sector(coin["id"], coin["name"])
        await conn.execute("""
            INSERT INTO assets (coin_id, symbol, name, sector, market_cap_rank, market_cap_usd, current_price_usd, price_change_24h, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (coin_id) DO UPDATE SET
                market_cap_rank = EXCLUDED.market_cap_rank,
                market_cap_usd = EXCLUDED.market_cap_usd,
                current_price_usd = EXCLUDED.current_price_usd,
                price_change_24h = EXCLUDED.price_change_24h,
                updated_at = NOW()
        """,
            coin["id"], coin["symbol"].upper(), coin["name"],
            sector, coin["market_cap_rank"],
            float(coin.get("market_cap") or 0),
            float(coin.get("current_price") or 0),
            float(coin.get("price_change_percentage_24h") or 0)
        )
        count += 1
    log.info(f"Synced {count} assets")
    return count


async def main():
    database_url = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://")
    conn = await asyncpg.connect(database_url)
    try:
        await init_assets_table(conn)
        await sync_assets(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
