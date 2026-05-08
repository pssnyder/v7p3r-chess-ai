"""Quick script to check BigQuery table schemas"""
from google.cloud import bigquery

PROJECT_ID = "chess-engine-metrics-agent"
client = bigquery.Client(project=PROJECT_ID)

print("Checking schema for conformed_layer.moves:")
print("=" * 60)

# Get table schema
table_ref = f"{PROJECT_ID}.conformed_layer.moves"
table = client.get_table(table_ref)

for field in table.schema:
    print(f"  {field.name:30} {field.field_type:15} {field.mode}")

print("\n\nChecking schema for conformed_layer.game_data:")
print("=" * 60)

table_ref = f"{PROJECT_ID}.conformed_layer.game_data"
table = client.get_table(table_ref)

for field in table.schema:
    print(f"  {field.name:30} {field.field_type:15} {field.mode}")

print("\n\nQuerying sample row from moves table:")
print("=" * 60)

query = f"""
SELECT *
FROM `{PROJECT_ID}.conformed_layer.moves`
LIMIT 1
"""

result = client.query(query).result()
for row in result:
    for key in row.keys():
        print(f"  {key}: {row[key]}")
