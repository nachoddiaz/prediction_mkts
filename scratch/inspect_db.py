import duckdb

con = duckdb.connect("data/duckdb/markets.duckdb")
print("Markets by Venue:")
print(con.execute("select venue, count(*) from markets group by venue").fetchall())
print("\nTicks by Venue (via market venue):")
print(
    con.execute(
        "select m.venue, count(*)"
        " from ticks t join markets m on t.market_id = m.market_id"
        " group by m.venue"
    ).fetchall()
)
print("\nSample Ticks:")
print(con.execute("select market_id, count(*) from ticks group by market_id").fetchall())
con.close()
