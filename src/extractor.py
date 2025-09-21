import orjson as json
import openai
import time
from datetime import datetime
from typing import Dict, List, Any, Optional


class MemoryExtractor:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract_facts(
        self,
        conversation: List[Dict[str, str]],
        existing_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract key facts from a conversation, handling both new facts and updates to existing facts.
        """
        # Format existing memories for context
        existing_context = ""
        if existing_memories:
            existing_context = "\n\nExisting memories:\n"
            for memory in existing_memories:
                existing_context += f"- {memory['content']}\n"

        # Format conversation for context
        conversation_text = ""
        for i, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            conversation_text += f"{role.capitalize()}: {content}\n"

        # Create the prompt for fact extraction
        prompt = f"""
        Analyze the following conversation and extract key facts about the user. 
        For each fact, determine if it's a new fact or an update to an existing fact.
        
        {existing_context}
        
        Conversation:
        {conversation_text}
        
        Please extract facts in JSON format with the following structure:
        {{
            "facts": [
                {{
                    "content": "The fact content",
                    "is_update": true/false,
                    "previous_value": "Previous fact content if it's an update, otherwise null",
                    "confidence": 0.95  # Confidence score between 0 and 1
                }}
            ]
        }}
        
        Only include facts that are clearly stated in the conversation.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            # Process the extracted facts and add metadata
            memories = []
            for i, fact in enumerate(result["facts"]):
                memory = {
                    "fact_id": f"f_{int(time.time())}_{i}",
                    "content": fact["content"],
                    "extracted_from": f"turn_{len(conversation)}",
                    "confidence": fact["confidence"],
                    "timestamp": datetime.now().isoformat(),
                    "is_update": fact["is_update"],
                }

                if (
                    fact["is_update"]
                    and "previous_value" in fact
                    and fact["previous_value"]
                ):
                    memory["previous_value"] = fact["previous_value"]

                memories.append(memory)

            return memories

        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []
