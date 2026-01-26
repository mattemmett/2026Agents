import os

print("Up")
print("DATABASE_URL =", os.getenv("DATABASE_URL"))
print("REDIS_URL    =", os.getenv("REDIS_URL"))
print("WEAVIATE_URL =", os.getenv("WEAVIATE_URL"))