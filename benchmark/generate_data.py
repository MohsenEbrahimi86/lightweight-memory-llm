import orjson as json
import openai
import random
from typing import List, Dict, Any


class ConversationGenerator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_conversations(
        self, num_conversations: int = 50, output_path: str = "data/conversations.json"
    ) -> List[Dict[str, Any]]:
        """Generate synthetic conversations for benchmarking"""
        conversations = []

        # Define scenarios
        scenarios = [
            "Simple fact storage and retrieval",
            "Fact updates and corrections",
            "Multi-hop reasoning, where facts relate to each other",
        ]

        for i in range(num_conversations):
            # Select a scenario
            scenario = random.choice(scenarios)

            # Create prompt for conversation generation
            prompt = f"""
            Generate a conversation between a user and an assistant that demonstrates {scenario}.
            
            The conversation should have 4-8 turns and include:
            - User sharing information about themselves
            - Assistant responding appropriately
            - Clear facts that can be extracted about the user
            
            Format the conversation as a JSON object with a "turns" array, where each turn has "role" and "content":
            
            {{
                "scenario": "{scenario}",
                "turns": [
                    {{"role": "user", "content": "..."}},
                    {{"role": "assistant", "content": "..."}},
                    ...
                ]
            }}
            
            Make the conversation natural and realistic.
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    response_format={"type": "json_object"},
                )

                conversation = json.loads(response.choices[0].message.content)
                conversations.append(conversation)

            except Exception as e:
                print(f"Error generating conversation {i}: {e}")
                # Add a simple fallback conversation
                conversations.append(
                    {
                        "scenario": scenario,
                        "turns": [
                            {
                                "role": "user",
                                "content": "My name is John and I live in New York.",
                            },
                            {
                                "role": "assistant",
                                "content": "Nice to meet you, John! How do you like living in New York?",
                            },
                        ],
                    }
                )

        # Save conversations to file
        with open(output_path, "w") as f:
            json.dump(conversations, f, indent=2)

        print(
            f"Generated {len(conversations)} conversations and saved to {output_path}"
        )
        return conversations

    def generate_expected_memories(
        self,
        conversations: List[Dict[str, Any]],
        output_path: str = "data/expected_memories.json",
    ) -> List[Dict[str, Any]]:
        """Generate expected memories for each conversation to use as ground truth"""
        all_expected_memories = []

        for i, conversation in enumerate(conversations):
            # Format conversation for context
            conversation_text = ""
            for turn in conversation["turns"]:
                conversation_text += f"{turn['role'].capitalize()}: {turn['content']}\n"

            # Create prompt for expected memory generation
            prompt = f"""
            Analyze the following conversation and identify the key facts about the user that should be stored in memory.
            
            Conversation:
            {conversation_text}
            
            Please provide the expected memories in JSON format with the following structure:
            {{
                "expected_memories": [
                    {{
                        "content": "The fact content",
                        "turn_extracted_from": 1  # Index of the turn where this fact appears
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

                result = json.loads(response.choices[0].message.content)

                # Add conversation index to each memory
                for memory in result["expected_memories"]:
                    memory["conversation_index"] = i
                    memory["scenario"] = conversation["scenario"]

                all_expected_memories.extend(result["expected_memories"])

            except Exception as e:
                print(f"Error generating expected memories for conversation {i}: {e}")

        # Save expected memories to file
        with open(output_path, "w") as f:
            json.dump(all_expected_memories, f, indent=2)

        print(
            f"Generated {len(all_expected_memories)} expected memories and saved to {output_path}"
        )
        return all_expected_memories
