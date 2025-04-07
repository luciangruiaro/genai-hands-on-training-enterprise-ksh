import json
import os

KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "agent_knowledge.json")


class KnowledgeManager:
    def __init__(self):
        self._knowledge = self._load_knowledge()

    def _load_knowledge(self):
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r") as f:
                return json.load(f)
        return {}

    def get_knowledge(self):
        self._knowledge = self._load_knowledge()
        return self._knowledge

    def update_knowledge(self, new_data: dict):
        for key, value in new_data.items():
            if isinstance(value, list):
                # Patch list of objects with 'name' as key
                if key in self._knowledge and isinstance(self._knowledge[key], list):
                    existing_list = self._knowledge[key]
                    updated_list = self._patch_list(existing_list, value)
                    self._knowledge[key] = updated_list
                else:
                    self._knowledge[key] = value
            elif isinstance(value, dict):
                if isinstance(self._knowledge.get(key), dict):
                    self._knowledge[key].update(value)
                else:
                    self._knowledge[key] = value
            else:
                self._knowledge[key] = value

        self._save_knowledge()

    def _patch_list(self, original_list, updates):
        result = original_list[:]
        for update_item in updates:
            if not isinstance(update_item, dict) or "name" not in update_item:
                continue
            for i, original_item in enumerate(result):
                if original_item.get("name") == update_item["name"]:
                    result[i].update(update_item)
                    break
            else:
                # If no match found, append new
                result.append(update_item)
        return result

    def _save_knowledge(self):
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(self._knowledge, f, indent=2)

    def refresh(self):
        self._knowledge = self._load_knowledge()


knowledge_curr = KnowledgeManager()
