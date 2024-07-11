from pymongo import MongoClient
from config import MONGO_URI

# Connection to MongoDB
client = MongoClient(MONGO_URI)
db = client["forestdb"]
users = db.users

# Fetching all documents
all_users = users.find()
for user in all_users:
    print(user)
