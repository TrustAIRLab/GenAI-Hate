#!/usr/bin/env python3
"""
Monthly stratified sampling for Pangram, up to N per month.
Skip posts already present in results_table.

Usage:
  python run_pangram_sampled.py \
      --source-table antisemitic_posts \
      --results-table pangram_antisemitic_results \
      --per-month 300 --workers 10
"""

import os
import sys
import json
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'data_prep'))
from clean_text_utils import clean_4chan_text

from pangram import Pangram

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_db_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', '4chan'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
    }


def get_monthly_sample(conn, source_table, results_table, per_month, min_cleaned_length):
    """Monthly random sampling, skipping posts with existing results."""
    posts = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get all months
        cur.execute(f"""
            SELECT DISTINCT to_char(to_timestamp(timestamp), 'YYYY-MM') as month
            FROM {source_table}
            ORDER BY month
        """)
        months = [r['month'] for r in cur.fetchall()]

        for month in months:
            # Check how many results exist for the month
            cur.execute(f"""
                SELECT COUNT(*) FROM {results_table} r
                JOIN {source_table} s ON r.post_num = s.num
                WHERE to_char(to_timestamp(s.timestamp), 'YYYY-MM') = %s
                  AND r.api_error = FALSE
            """, (month,))
            existing = cur.fetchone()['count']

            if existing >= per_month:
                logger.info(f"  {month}: already has {existing} results, skipping")
                continue

            need = per_month - existing

            # Randomly sample untested posts
            cur.execute(f"""
                SELECT s.num, s.comment, s.timestamp
                FROM {source_table} s
                LEFT JOIN {results_table} r ON s.num = r.post_num
                WHERE to_char(to_timestamp(s.timestamp), 'YYYY-MM') = %s
                  AND r.post_num IS NULL
                ORDER BY random()
                LIMIT %s
            """, (month, need))
            month_posts = cur.fetchall()

            # Clean filter
            cleaned = []
            for p in month_posts:
                text = clean_4chan_text(p['comment'], remove_quote_markers=True)
                if len(text) >= min_cleaned_length:
                    cleaned.append({
                        'num': p['num'],
                        'timestamp': p['timestamp'],
                        'cleaned': text,
                    })

            posts.extend(cleaned)
            logger.info(f"  {month}: need {need}, sampled {len(cleaned)}")

    return posts


def save_result(conn, results_table, post_num, timestamp, cleaned_text, result, error_msg):
    with conn.cursor() as cur:
        if error_msg or result is None:
            cur.execute(f"""
                INSERT INTO {results_table}
                (post_num, timestamp, cleaned_text, cleaned_length, api_error, error_message)
                VALUES (%s, %s, %s, %s, TRUE, %s)
                ON CONFLICT (post_num) DO UPDATE SET
                    api_error = TRUE, error_message = EXCLUDED.error_message,
                    tested_at = CURRENT_TIMESTAMP
            """, (post_num, timestamp, cleaned_text, len(cleaned_text) if cleaned_text else 0,
                  error_msg or 'Unknown error'))
        else:
            cur.execute(f"""
                INSERT INTO {results_table}
                (post_num, timestamp, cleaned_text, cleaned_length,
                 ai_likelihood, prediction_short, headline,
                 fraction_ai, fraction_ai_assisted, fraction_human,
                 num_ai_segments, num_ai_assisted_segments, num_human_segments,
                 version, full_response, api_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (post_num) DO UPDATE SET
                    cleaned_text = EXCLUDED.cleaned_text, cleaned_length = EXCLUDED.cleaned_length,
                    ai_likelihood = EXCLUDED.ai_likelihood, prediction_short = EXCLUDED.prediction_short,
                    headline = EXCLUDED.headline, fraction_ai = EXCLUDED.fraction_ai,
                    fraction_ai_assisted = EXCLUDED.fraction_ai_assisted,
                    fraction_human = EXCLUDED.fraction_human,
                    num_ai_segments = EXCLUDED.num_ai_segments,
                    num_ai_assisted_segments = EXCLUDED.num_ai_assisted_segments,
                    num_human_segments = EXCLUDED.num_human_segments,
                    version = EXCLUDED.version, full_response = EXCLUDED.full_response,
                    api_error = FALSE, error_message = NULL, tested_at = CURRENT_TIMESTAMP
            """, (
                post_num, timestamp, cleaned_text, len(cleaned_text),
                result.get('ai_likelihood'), result.get('prediction_short'),
                result.get('headline'), result.get('fraction_ai'),
                result.get('fraction_ai_assisted'), result.get('fraction_human'),
                result.get('num_ai_segments'), result.get('num_ai_assisted_segments'),
                result.get('num_human_segments'), result.get('version'), json.dumps(result),
            ))
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Monthly stratified Pangram detection')
    parser.add_argument('--db-config', type=str, default='../db_config.json')
    parser.add_argument('--source-table', type=str, required=True)
    parser.add_argument('--results-table', type=str, required=True)
    parser.add_argument('--per-month', type=int, default=300)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--min-cleaned-length', type=int, default=50)
    args = parser.parse_args()

    # Validate API key
    api_key = os.environ.get('PANGRAM_API_KEY', '')
    if not api_key:
        logger.error("PANGRAM_API_KEY not found! Check pangram/.env")
        sys.exit(1)
    logger.info(f"API key loaded: {api_key[:8]}...")

    # Test one sample to validate API
    client = Pangram()
    try:
        test_result = client.predict("This is a test sentence to verify the API connection works.")
        assert 'fraction_ai' in test_result, f"Unexpected API response: {test_result.keys()}"
        logger.info(f"API test OK (fraction_ai={test_result['fraction_ai']})")
    except Exception as e:
        logger.error(f"API test FAILED: {e}")
        sys.exit(1)

    db_config = load_db_config(args.db_config)
    conn = psycopg2.connect(**db_config)

    # Ensure results table exists
    from test_pangram import create_results_table
    create_results_table(conn, args.results_table)

    # Monthly sampling
    logger.info(f"Monthly sampling: {args.source_table} -> {args.results_table}, {args.per_month}/month")
    posts = get_monthly_sample(conn, args.source_table, args.results_table,
                                args.per_month, args.min_cleaned_length)
    logger.info(f"Total to detect: {len(posts)} posts")

    if not posts:
        logger.info("No posts to detect")
        conn.close()
        return

    # Parallel detection
    success = 0
    errors = 0
    lock = threading.Lock()

    def process(post):
        try:
            result = client.predict(post['cleaned'])
            return post, result, None
        except Exception as e:
            return post, None, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, p): p for p in posts}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pangram"):
            post, result, err = future.result()
            with lock:
                save_result(conn, args.results_table, post['num'], post['timestamp'],
                           post['cleaned'], result, err)
                if err:
                    errors += 1
                else:
                    success += 1

    logger.info(f"Done: success {success}, errors {errors}")
    conn.close()


if __name__ == '__main__':
    main()
