import duckdb
import time

con = duckdb.connect()

# ── 1. Create a 2M row table ──────────────────────────────────────────────────
print("Setting up 2M row orders table...")
con.execute("""
    CREATE TABLE orders AS
    SELECT
        (random() * 1_000_000)::INT                                        AS customer_id,
        (random() * 500)::INT                                              AS product_id,
        (DATE '2022-01-01' + ((random() * 730)::INT || ' days')::INTERVAL) AS order_date,
        round(random() * 500 + 5, 2)                                       AS amount
    FROM range(2_000_000)
""")
print("  Done — 2,000,000 rows loaded\n")

# ── 2. Query WITHOUT index ────────────────────────────────────────────────────
print("=" * 60)
print("EXPERIMENT A: Query WITHOUT an index")
print("=" * 60)

t0 = time.perf_counter()
result_no_idx = con.execute("""
    SELECT COUNT(*), round(AVG(amount), 2) AS avg_spend
    FROM orders WHERE customer_id = 42765
""").fetchone()
t1 = time.perf_counter()
no_idx_ms = (t1 - t0) * 1000

print(f"  Result : {result_no_idx}")
print(f"  Time   : {no_idx_ms:.1f} ms")

plan_no_idx = con.execute("""
    EXPLAIN SELECT COUNT(*), round(AVG(amount), 2)
    FROM orders WHERE customer_id = 42765
""").fetchall()
print("\n  Query plan (no index):")
for row in plan_no_idx:
    print(f"    {row[1][:120]}")

# ── 3. Create index ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Creating index on customer_id...")
con.execute("CREATE INDEX idx_customer ON orders(customer_id)")
print("  Index created ✓")

# ── 4. Query WITH index ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT B: Same query WITH an index")
print("=" * 60)

t2 = time.perf_counter()
result_idx = con.execute("""
    SELECT COUNT(*), round(AVG(amount), 2) AS avg_spend
    FROM orders WHERE customer_id = 42765
""").fetchone()
t3 = time.perf_counter()
idx_ms = (t3 - t2) * 1000

print(f"  Result : {result_idx}")
print(f"  Time   : {idx_ms:.1f} ms")

plan_idx = con.execute("""
    EXPLAIN SELECT COUNT(*), round(AVG(amount), 2)
    FROM orders WHERE customer_id = 42765
""").fetchall()
print("\n  Query plan (with index):")
for row in plan_idx:
    print(f"    {row[1][:120]}")

# ── 5. The trap — low cardinality ────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT C: The trap — index on a LOW cardinality column")
print("  product_id has only ~500 unique values across 2M rows")
print("=" * 60)

con.execute("CREATE INDEX idx_product ON orders(product_id)")

t4 = time.perf_counter()
con.execute("""
    SELECT COUNT(*), round(AVG(amount), 2)
    FROM orders WHERE product_id = 42
""").fetchone()
t5 = time.perf_counter()
low_card_ms = (t5 - t4) * 1000

print(f"  Time with index on product_id : {low_card_ms:.1f} ms")
print("  → Each product_id matches ~4000 rows.")
print("    A full columnar scan can be cheaper than scattered index lookups!")

# ── 6. Summary ────────────────────────────────────────────────────────────────
improvement = no_idx_ms / idx_ms if idx_ms > 0 else 0

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Without index (full scan)       : {no_idx_ms:.1f} ms")
print(f"  With index on customer_id       : {idx_ms:.1f} ms  ← {improvement:.1f}x faster")
print(f"  With index on product_id (trap) : {low_card_ms:.1f} ms")
print()
print("KEY LESSONS FOR YOUR ARTICLE:")
print("  1. Index = sorted pointer list — seek, not scan")
print("  2. High cardinality columns (lots of unique values) benefit most")
print("  3. Low cardinality columns — the index can be ignored entirely")
print("  4. EXPLAIN shows what the DB actually decided — always check it")
print("  5. Interview answer: 'It trades write overhead for read speed,")
print("     but only when selectivity is high enough to be worth it.'")
