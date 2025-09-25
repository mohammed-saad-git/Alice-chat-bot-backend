import json
import os

class UserData:
    def __init__(self, filename="user_data.json"):
        self.filename = filename
        self.data = self.load_data()
    
    def load_data(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_data(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_user_data(self, user_id, key=None):
        if user_id not in self.data:
            self.data[user_id] = {}
        
        if key:
            return self.data[user_id].get(key)
        return self.data[user_id]
    
    def set_user_data(self, user_id, key, value):
        if user_id not in self.data:
            self.data[user_id] = {}
        
        self.data[user_id][key] = value
        self.save_data()
    
    def delete_user_data(self, user_id, key=None):
        if user_id in self.data:
            if key:
                if key in self.data[user_id]:
                    del self.data[user_id][key]
                    self.save_data()
            else:
                del self.data[user_id]
                self.save_data()

# Create a global instance
user_db = UserData()