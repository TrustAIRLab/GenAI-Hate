"""
Optimized batch processing for large datasets with pagination and resume support

This script is designed for processing millions of comments efficiently:
- Pagination: Query database in small batches to avoid memory issues
- Resume support: Skip already analyzed comments
- Progress tracking: Save progress periodically
- Index optimization: Use efficient queries for large tables

Usage:
    # Process comments in batches of 100
    python query_and_analyze_batch.py \
      --keywords "hate" "racist" \
      --batch-size 100 \
      --max-total 10000

    # Resume from previous run
    python query_and_analyze_batch.py \
      --keywords "hate" \
      --resume
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from config import DELAY_BETWEEN_REQUESTS

# Load environment variables
load_dotenv()

# Default configuration
COMMENT_TABLE = os.getenv('COMMENT_TABLE', 'pol_posts_2026')
COMMENT_COLUMN = os.getenv('COMMENT_COLUMN', 'comment')
TIME_COLUMN = os.getenv('TIME_COLUMN', 'timestamp')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer')
}

# Default attributes
DEFAULT_ANALYSIS_ATTRIBUTES = ['SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT']

# Batch processing settings
DEFAULT_BATCH_SIZE = 100  # Process 100 comments at a time
PROGRESS_SAVE_INTERVAL = 10  # Save progress every 10 batches


class PerspectiveAPIClient:
    """Minimal Google Perspective API client (embedded to avoid extra imports)."""
    ATTRIBUTES = [
        TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT,
        PROFANITY, THREAT, SEXUALLY_EXPLICIT, FLIRTATION
    ]

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(PERSPECTIVE_API_KEY)
        if not api_key:
            raise ValueError("PERSPECTIVE_API_KEY not set")
        self.client = discovery.build(
            "commentanalyzer", "v1alpha1", developerKey=api_key,
            discoveryServiceUrl=(
                "https://commentanalyzer.googleapis.com//rest?version=v1alpha1"
            ),
            static_discovery=False,
        )

    def analyze_comment(self, text: str, attributes: Optional[List[str]] = None,
                        languages: Optional[List[str]] = None) -> Dict:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        attributes = attributes or [TOXICITY]
        languages = languages or [en]
        body = {
            comment: {text: text},
            requestedAttributes: {a: {} for a in attributes},
            languages: languages,
        }
        try:
            resp = self.client.comments().analyze(body=body).execute()
        except HttpError as e:
            raise RuntimeError(f"Perspective API error: {e}")
        out = {}
        for k, v in (resp.get(attributeScores) or {}).items():
            s = v.get(summaryScore, {})
            out[k] = {score: s.get(value, 0.0), type: s.get(type, PROBABILITY)}
        return out

    def analyze_batch(self, texts: List[str], attributes: Optional[List[str]] = None,
                      delay: float = 1.0) -> List[Dict]:
        results = []
        for i, t in enumerate(texts):
            try:
                scores = self.analyze_comment(t, attributes)
                results.append({index: i, text: t, scores: scores, error: None})
                if i < len(texts) - 1:
                    time.sleep(delay)
            except Exception as e:
                results.append({index: i, text: t, scores: None, error: str(e)})
        return results


class BatchCommentAnalyzer:
    """Optimized analyzer for large datasets with pagination and resume support"""

    def __init__(self, db_config: Dict = None, perspective_api_key: str = None):
        """Initialize analyzer"""
        self.db_config = db_config or DB_CONFIG
        self.perspective_client = PerspectiveAPIClient(api_key=perspective_api_key)
        self.conn = None

        # Progress tracking
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.start_time = time.time()

    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print(f"✅ Connected to database: {self.db_config['database']}")

            # Force use of indexes (disable sequential scan)
            # This is necessary when query planner incorrectly chooses seq scan
            with self.conn.cursor() as cur:
                cur.execute("SET enable_seqscan = off;")
            print("✅ Forced index usage (disabled seq scan)")

        except Exception as e:
            raise Exception(f"Failed to connect to database: {e}")

    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def _create_results_table(self, attributes: List[str]):
        """Create perspective_results table if not exists"""
        if not self.conn:
            self.connect_db()

        print("Creating perspective_results table if not exists...")
        score_columns = [f"{attr.lower()} FLOAT" for attr in attributes]

        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS perspective_results (
                id SERIAL PRIMARY KEY,
                num BIGINT NOT NULL,
                subnum INTEGER NOT NULL,
                comment_text TEXT NOT NULL,
                comment_timestamp BIGINT,
                {', '.join(score_columns)},
                analyzed_at TIMESTAMP NOT NULL,
                CONSTRAINT unique_comment UNIQUE (num, subnum)
            );

            CREATE INDEX IF NOT EXISTS idx_perspective_num_subnum
                ON perspective_results(num, subnum);
            CREATE INDEX IF NOT EXISTS idx_perspective_analyzed_at
                ON perspective_results(analyzed_at);
        """

        for attr in attributes:
            create_table_query += f"""
            CREATE INDEX IF NOT EXISTS idx_perspective_{attr.lower()}
                ON perspective_results({attr.lower()});
            """

        with self.conn.cursor() as cur:
            cur.execute(create_table_query)
            self.conn.commit()

        print("✅ Table and indexes ready")

    def count_total_matches(
        self,
        keywords: List[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> int:
        """
        Count total number of matching comments (for progress display)

        NOTE: This might be slow on large tables. Use with caution.
        """
        if not self.conn:
            self.connect_db()

        keyword_conditions = " OR ".join([f"{COMMENT_COLUMN} ILIKE %s" for _ in keywords])
        keyword_params = [f"%{kw}%" for kw in keywords]

        where_clauses = [f"({keyword_conditions})"]
        params = keyword_params

        if start_time:
            start_ts = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp())
            where_clauses.append(f"{TIME_COLUMN} >= %s")
            params.append(start_ts)

        if end_time:
            end_ts = int(datetime.strptime(end_time + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp())
            where_clauses.append(f"{TIME_COLUMN} <= %s")
            params.append(end_ts)

        where_sql = " AND ".join(where_clauses)

        # Use COUNT(*) - might be slow on large tables
        query = f"SELECT COUNT(*) FROM {COMMENT_TABLE} WHERE {where_sql}"

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            count = cur.fetchone()[0]

        return count

    def query_comments_batch(
        self,
        keywords: List[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        offset: int = 0,
        skip_analyzed: bool = True,
        min_char_length: int = 0,
        exclude_keywords: List[str] = None,
    ) -> Tuple[List[Dict], bool]:
        """
        Query comments in batches using pagination

        Args:
            keywords: List of keywords to search for (use ["__RANDOM__"] for random baseline)
            start_time: Start time (YYYY-MM-DD)
            end_time: End time (YYYY-MM-DD)
            batch_size: Number of comments per batch
            offset: Offset for pagination
            skip_analyzed: Skip comments already in perspective_results
            exclude_keywords: Keywords to exclude (for baseline: exclude target words)

        Returns:
            Tuple of (comments, has_more)
        """
        if not self.conn:
            self.connect_db()

        where_clauses = []
        params = []

        # Random baseline mode: no keyword filter, use TABLESAMPLE
        is_random = keywords == ["__RANDOM__"]

        if not is_random:
            keyword_conditions = " OR ".join([f"{COMMENT_COLUMN} ILIKE %s" for _ in keywords])
            keyword_params = [f"%{kw}%" for kw in keywords]
            where_clauses.append(f"({keyword_conditions})")
            params = keyword_params

        if start_time:
            start_ts = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp())
            where_clauses.append(f"{TIME_COLUMN} >= %s")
            params.append(start_ts)

        if end_time:
            end_ts = int(datetime.strptime(end_time + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp())
            where_clauses.append(f"{TIME_COLUMN} <= %s")
            params.append(end_ts)

        # Minimum character length filter
        if min_char_length > 0:
            where_clauses.append(f"char_length({COMMENT_COLUMN}) >= %s")
            params.append(min_char_length)

        # Exclude keywords (for baseline: exclude "india", "jews", etc.)
        if exclude_keywords:
            for ek in exclude_keywords:
                where_clauses.append(f"{COMMENT_COLUMN} NOT ILIKE %s")
                params.append(f"%{ek}%")

        # Comment must not be null/empty
        where_clauses.append(f"{COMMENT_COLUMN} IS NOT NULL")
        where_clauses.append(f"char_length({COMMENT_COLUMN}) > 10")

        # Skip already analyzed comments
        if skip_analyzed:
            where_clauses.append(
                f"NOT EXISTS (SELECT 1 FROM perspective_results pr "
                f"WHERE pr.num = {COMMENT_TABLE}.num AND pr.subnum = {COMMENT_TABLE}.subnum)"
            )

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

        if is_random:
            # Random sampling using TABLESAMPLE + ORDER BY RANDOM()
            query = f"""
                SELECT num, subnum, {COMMENT_COLUMN}, {TIME_COLUMN}
                FROM {COMMENT_TABLE} TABLESAMPLE SYSTEM(0.01)
                WHERE {where_sql}
                ORDER BY RANDOM()
                LIMIT %s
            """
            params.append(batch_size)
        else:
            # Standard keyword-based pagination
            query = f"""
                SELECT num, subnum, {COMMENT_COLUMN}, {TIME_COLUMN}
                FROM {COMMENT_TABLE}
                WHERE {where_sql}
                ORDER BY {TIME_COLUMN} DESC
                LIMIT %s OFFSET %s
            """
            params.extend([batch_size + 1, offset])

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()

        if is_random:
            has_more = len(results) == batch_size  # assume more if we got a full batch
        else:
            has_more = len(results) > batch_size
            if has_more:
                results = results[:batch_size]

        return results, has_more

    def analyze_and_save_batch(
        self,
        comments: List[Dict],
        attributes: List[str],
        delay: float
    ) -> Dict:
        """
        Analyze a batch of comments and save to database immediately

        Returns:
            Dictionary with statistics
        """
        if not comments:
            return {'successful': 0, 'failed': 0}

        # Extract texts
        texts = [comment[COMMENT_COLUMN] for comment in comments]

        # Analyze batch
        results = self.perspective_client.analyze_batch(
            texts=texts,
            attributes=attributes,
            delay=delay
        )

        # Add identifiers
        for i, result in enumerate(results):
            result['num'] = comments[i]['num']
            result['subnum'] = comments[i]['subnum']
            result['timestamp'] = comments[i]['timestamp']

        # Save to database immediately
        insert_data = []
        successful = 0
        failed = 0

        for result in results:
            if result['error'] is not None:
                failed += 1
                continue

            row = {
                'num': result['num'],
                'subnum': result['subnum'],
                'comment_text': result['text'],
                'comment_timestamp': result['timestamp'],
                'analyzed_at': datetime.utcnow()
            }

            for attr_name in attributes:
                if attr_name in result['scores']:
                    score = result['scores'][attr_name]['score']
                    row[attr_name.lower()] = score

            insert_data.append(row)
            successful += 1

        if insert_data:
            columns = list(insert_data[0].keys())
            values_template = f"({', '.join(['%s'] * len(columns))})"

            insert_query = f"""
                INSERT INTO perspective_results ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (num, subnum) DO UPDATE SET
                    {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['num', 'subnum', 'comment_text', 'comment_timestamp']])}
            """

            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    insert_query,
                    [tuple(row[col] for col in columns) for row in insert_data],
                    template=values_template
                )
                self.conn.commit()

        return {'successful': successful, 'failed': failed}

    def process_all_batches(
        self,
        keywords: List[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_total: Optional[int] = None,
        delay: float = None,
        skip_analyzed: bool = True,
        min_char_length: int = 0,
        exclude_keywords: List[str] = None,
    ):
        """
        Process all matching comments in batches

        Args:
            keywords: Keywords to search for
            start_time: Start time filter
            end_time: End time filter
            attributes: Attributes to analyze
            batch_size: Comments per batch
            max_total: Maximum total comments to process (None = unlimited)
            delay: Delay between API calls
            skip_analyzed: Skip already analyzed comments
        """
        if attributes is None:
            attributes = DEFAULT_ANALYSIS_ATTRIBUTES

        if delay is None:
            delay = DELAY_BETWEEN_REQUESTS

        # Create results table
        self._create_results_table(attributes)

        print("\n" + "="*70)
        print("BATCH PROCESSING MODE")
        print("="*70)
        print(f"Keywords: {keywords}")
        print(f"Batch size: {batch_size}")
        print(f"Attributes: {', '.join(attributes)}")
        print(f"Skip analyzed: {skip_analyzed}")
        if max_total:
            print(f"Max total: {max_total}")
        print("="*70 + "\n")

        offset = 0
        batch_num = 0
        has_more = True

        while has_more:
            batch_num += 1

            # Check if we've reached max_total
            if max_total and self.total_processed >= max_total:
                print(f"\n✅ Reached maximum total limit ({max_total})")
                break

            print(f"\n📦 Batch {batch_num} (offset: {offset})")
            print("-" * 70)

            # Query batch
            print(f"Querying database for batch {batch_num}...")
            comments, has_more = self.query_comments_batch(
                keywords=keywords,
                start_time=start_time,
                end_time=end_time,
                batch_size=batch_size,
                offset=offset,
                skip_analyzed=skip_analyzed,
                min_char_length=min_char_length,
                exclude_keywords=exclude_keywords,
            )

            if not comments:
                print("No more comments to process")
                break

            print(f"Fetched {len(comments)} comments from database")

            # Analyze and save
            stats = self.analyze_and_save_batch(comments, attributes, delay)

            # Update totals
            self.total_processed += len(comments)
            self.total_successful += stats['successful']
            self.total_failed += stats['failed']

            # Progress report
            elapsed = time.time() - self.start_time
            rate = self.total_processed / elapsed if elapsed > 0 else 0

            print(f"✅ Batch complete: {stats['successful']} successful, {stats['failed']} failed")
            print(f"📊 Total progress: {self.total_processed} processed, "
                  f"{self.total_successful} successful, {self.total_failed} failed")
            print(f"⏱️  Rate: {rate:.2f} comments/sec, Elapsed: {elapsed/60:.1f} min")

            # Move to next batch
            if not skip_analyzed:
                offset += batch_size

        # Final summary
        elapsed = time.time() - self.start_time
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total processed: {self.total_processed}")
        print(f"Successful: {self.total_successful}")
        print(f"Failed: {self.total_failed}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Average rate: {self.total_processed/elapsed:.2f} comments/sec")
        print("="*70 + "\n")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch process comments with pagination and resume support"
    )

    # Query parameters
    parser.add_argument(
        '--keywords',
        nargs='+',
        help='Keywords to search for in comments'
    )
    parser.add_argument(
        '--random-baseline',
        action='store_true',
        help='Random baseline mode: sample random posts (no keyword filter)'
    )
    parser.add_argument(
        '--exclude-keywords',
        nargs='+',
        help='Exclude posts containing these keywords (for baseline: exclude target words)'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start time (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End time (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--min-char-length',
        type=int,
        default=0,
        help='Minimum character length for comments (default: 0, no filter)'
    )

    # Batch processing parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Number of comments per batch (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--max-total',
        type=int,
        help='Maximum total comments to process (default: unlimited)'
    )
    parser.add_argument(
        '--no-skip-analyzed',
        action='store_true',
        help='Do not skip already analyzed comments'
    )

    # Analysis parameters
    parser.add_argument(
        '--attributes',
        nargs='+',
        default=DEFAULT_ANALYSIS_ATTRIBUTES,
        choices=PerspectiveAPIClient.ATTRIBUTES,
        help='Attributes to analyze'
    )
    parser.add_argument(
        '--delay',
        type=float,
        help=f'Delay between API requests in seconds (default: {DELAY_BETWEEN_REQUESTS})'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Determine keywords
    if args.random_baseline:
        keywords = ["__RANDOM__"]
        mode_str = "RANDOM BASELINE"
    elif args.keywords:
        keywords = args.keywords
        mode_str = "KEYWORD SEARCH"
    else:
        print("Error: must specify --keywords or --random-baseline")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print(f"BATCH COMMENT ANALYSIS - {mode_str}")
    print("=" * 80)
    print("\n📋 Query Parameters:")
    if args.random_baseline:
        print(f"   Mode: Random baseline sampling")
        if args.exclude_keywords:
            print(f"   Excluding: {', '.join(args.exclude_keywords)}")
    else:
        print(f"   Keywords: {', '.join(keywords)}")
    print(f"   Time Range: {args.start or 'any'} to {args.end or 'any'}")
    print("\n⚙️  Batch Processing:")
    print(f"   Batch Size: {args.batch_size} comments per batch")
    print(f"   Max Total: {args.max_total or 'unlimited'}")
    print(f"   Skip Analyzed: {not args.no_skip_analyzed}")
    print("\n🔍 Analysis Settings:")
    print(f"   Attributes: {', '.join(args.attributes)}")
    print(f"   API Delay: {args.delay if args.delay else DELAY_BETWEEN_REQUESTS}s per request")
    print("\n" + "=" * 80 + "\n")

    try:
        # Initialize analyzer
        analyzer = BatchCommentAnalyzer()

        # Process all batches
        analyzer.process_all_batches(
            keywords=keywords,
            start_time=args.start,
            end_time=args.end,
            attributes=args.attributes,
            batch_size=args.batch_size,
            max_total=args.max_total,
            delay=args.delay,
            skip_analyzed=not args.no_skip_analyzed,
            min_char_length=args.min_char_length,
            exclude_keywords=args.exclude_keywords,
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print(f"Progress: {analyzer.total_processed} comments processed")
        print("You can resume by running the script again with the same parameters")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if hasattr(analyzer, 'conn'):
            analyzer.close_db()


if __name__ == "__main__":
    main()
