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
        if existing_memories:
            memory_lines = [f"- {memory['content']}" for memory in existing_memories]
            existing_context = "\n\nExisting memories:\n" + "\n".join(memory_lines)
        else:
            existing_context = ""

        # Format conversation for context
        if conversation:
            lines = [
                f"{turn.get('role', '').capitalize()}: {turn.get('content', '')}"
                for turn in conversation
            ]
            conversation_text = "\n".join(lines) + "\n"
        else:
            conversation_text = ""

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
            current_time = int(time.time())
            current_turn = f"turn_{len(conversation)}"
            now_iso = datetime.now().isoformat()

            memories = [
                {
                    "fact_id": f"f_{current_time}_{i}",
                    "content": fact["content"],
                    "extracted_from": current_turn,
                    "confidence": fact["confidence"],
                    "timestamp": now_iso,
                    "is_update": fact["is_update"],
                    **(
                        {"previous_value": fact["previous_value"]}
                        if fact.get("is_update") and fact.get("previous_value")
                        else {}
                    ),
                }
                for i, fact in enumerate(result["facts"])
            ]

            return memories

        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []
