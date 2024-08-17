import asyncio
import random
import json
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import os
import logging

# Set up logging
logging.basicConfig(filename='disco_simulation.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Initialize the Ollama client
client = AsyncOpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

@dataclass
class Agent:
    id: int
    preferred_style: str
    model: str
    partner_preferences: Dict[int, float] = field(default_factory=dict)

@dataclass
class DancePair:
    agent1: Agent
    agent2: Agent

@dataclass
class DanceResult:
    pair: DancePair
    routine: str
    score: int
    feedback: str
    iteration: int
    song: Dict[str, str]

class MultiAgentDisco:
    def __init__(self, num_agents: int, dance_styles: List[str], models: List[str], iterations: int, stabilization_threshold: int):
        self.num_agents = num_agents
        self.dance_styles = dance_styles
        self.models = models
        self.iterations = iterations
        self.stabilization_threshold = stabilization_threshold
        self.agents = self.initialize_agents()
        self.dance_history = []

    def initialize_agents(self) -> List[Agent]:
        return [Agent(id=i, preferred_style=random.choice(self.dance_styles), model=random.choice(self.models)) for i in range(self.num_agents)]

    async def retry_llm_request(self, func, *args, max_retries=10):
        for attempt in range(max_retries):
            try:
                return await func(*args)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in LLM request (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                logging.error(f"Error in LLM request (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise

    async def generate_song(self, agent: Agent) -> Dict[str, str]:
        prompt = "Generate a brief description for a dance song, including its title, genre, tempo, mood, and a short description."
        
        async def _generate():
            response = await client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": """You are a music composer AI assistant. Your task is to generate a JSON object representing musical composition parameters including title, genre, tempo, mood, and description.

    Follow these instructions to generate the JSON object:

    1. Create a JSON object with five key-value pairs: "title", "genre", "tempo", "mood", and "description".

    2. Use the following input variables to set the values:
    <title>{{TITLE}}</title>
    <genre>{{GENRE}}</genre>
    <tempo>{{TEMPO}}</tempo>
    <mood>{{MOOD}}</mood>
    <description>{{DESCRIPTION}}</description>

    3. If any of the input variables are empty or not provided, use the following default values:
    - For title: "Untitled"
    - For genre: "Default"
    - For tempo: "Moderate"
    - For mood: "Neutral"
    - For description: "A generic dance song"

    4. Ensure that all values are strings, even if they contain numbers.

    5. Format your response as a valid JSON object, enclosed in <response> tags.

    Here are examples of valid responses:

    <response>
    {"title": "Electric Dreams", "genre": "Electronic", "tempo": "Fast", "mood": "Energetic", "description": "A pulsating electronic dance track with synthesizer melodies and a driving beat"}
    </response>

    <response>
    {"title": "Moonlight Serenade", "genre": "Classical", "tempo": "Slow", "mood": "Melancholic", "description": "A gentle classical piece featuring soft piano and string arrangements"}
    </response>

    Remember to use the provided input variables if they are available, and only use the default values when necessary."""},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        try:
            song = await self.retry_llm_request(_generate)
            # Ensure all required keys are present
            return {
                "title": song.get("title", "Untitled"),
                "genre": song.get("genre", "Default"),
                "tempo": song.get("tempo", "Moderate"),
                "mood": song.get("mood", "Neutral"),
                "description": song.get("description", "A generic dance song")
            }
        except Exception as e:
            logging.error(f"Failed to generate song after all retries: {e}")
            return {
                "title": "Untitled",
                "genre": "Default",
                "tempo": "Moderate",
                "mood": "Neutral",
                "description": "A generic dance song"
            }

    def select_partners(self) -> List[DancePair]:
        available_agents = self.agents.copy()
        pairs = []
        while len(available_agents) >= 2:
            agent1 = random.choice(available_agents)
            available_agents.remove(agent1)
            agent2 = max(available_agents, key=lambda a: agent1.partner_preferences.get(a.id, 0))
            available_agents.remove(agent2)
            pairs.append(DancePair(agent1, agent2))
        return pairs

    async def collaborate_on_dance(self, pair: DancePair, song: Dict[str, str]) -> str:
        routine = []
        dancers = [pair.agent1, pair.agent2]
        
        async def _generate_move(dancer, turn):
            prompt = f"""
            Song: {json.dumps(song)}
            Dancer {dancer.id} ({dancer.preferred_style}) is performing.
            Previous moves: {' '.join(routine)}
            Describe the next dance move:
            """
            response = await client.chat.completions.create(
                model=dancer.model,
                messages=[
                    {"role": "system", "content": """You are an AI assistant tasked with generating a dance move for a dancer in response to a song and previous moves. Your goal is to create a cohesive and engaging dance routine by suggesting the next appropriate move.

Here's the information about the song:
<song>
{{SONG}}
</song>

Now, let's focus on the dancer and their previous moves:
Dancer ID: {{DANCER_ID}}
Preferred dance style: {{PREFERRED_STYLE}}
Previous moves: {{PREVIOUS_MOVES}}

To generate the next dance move, follow these steps:
1. Consider the song's rhythm, tempo, and mood.
2. Take into account the dancer's preferred style.
3. Analyze the previous moves to ensure continuity and flow in the routine.
4. Create a dance move that complements the existing routine and fits the song.

Your response should be a JSON object containing a single dance move. The move should be descriptive but concise, focusing on the key elements of the movement.

Format your response as follows:
<response>
{
  "move": "Your dance move description here"
}
</response>

Remember to make the move appropriate for the song, the dancer's style, and the flow of the routine. Be creative while maintaining coherence with the previous moves."""},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("move", "Unspecified move")

        for turn in range(4):  # 4 turns for a back-and-forth routine
            dancer = dancers[turn % 2]
            try:
                move = await self.retry_llm_request(_generate_move, dancer, turn)
                routine.append(f"Dancer {dancer.id}: {move}")
            except Exception as e:
                logging.error(f"Failed to generate move after all retries for Dancer {dancer.id} on turn {turn}: {e}")
                routine.append(f"Dancer {dancer.id}: [Move generation failed]")

        return "\n".join(routine)

    async def judge_dance(self, routine: str, agent: Agent) -> Tuple[int, str]:
        prompt = f"Judge this dance routine on a scale of 1 to 10 (use only integers) and provide brief feedback:\n{routine}"
        
        async def _judge():
            response = await client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": """You are a professional dance judge tasked with evaluating a dance routine. Your goal is to provide a fair and constructive assessment of the performance.

Here is the dance routine you need to judge:
<routine>
{{ROUTINE}}
</routine>

Carefully evaluate the routine, considering factors such as technique, creativity, musicality, and overall performance quality. Pay attention to both strengths and areas for improvement.

Before scoring, think about your assessment and prepare your feedback. Consider specific elements of the performance that stood out, both positively and negatively. Your feedback should be constructive, highlighting what was done well and suggesting areas for improvement.

After formulating your thoughts, assign a score to the routine on a scale of 1 to 10, using only integer values. A score of 1 represents a poor performance, while 10 represents an exceptional, flawless performance.

Provide your evaluation in the form of a JSON object with two keys: "score" and "feedback". The "score" should be an integer between 1 and 10, and the "feedback" should be a string containing your brief, constructive feedback about the routine.

Your response should be structured as follows:
<answer>
{
  "score": [Your integer score between 1 and 10],
  "feedback": "[Your brief, constructive feedback here]"
}
</answer>

Ensure that your JSON is properly formatted and that the "score" is an integer, not a string or float."""},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            score = int(result.get("score", 5))  # Ensure score is an integer
            feedback = result.get("feedback", "No feedback provided")
            return score, feedback

        try:
            score, feedback = await self.retry_llm_request(_judge)
            return score, feedback.strip()
        except Exception as e:
            logging.error(f"Error in judge_dance: {e}")
            return 5, "Default feedback due to judging error."

    def update_preferences(self, results: List[DanceResult]):
        for result in results:
            a1, a2 = result.pair.agent1, result.pair.agent2
            a1.partner_preferences[a2.id] = a1.partner_preferences.get(a2.id, 0) + result.score
            a2.partner_preferences[a1.id] = a2.partner_preferences.get(a1.id, 0) + result.score

    def calculate_highest_ranked_pairings(self) -> List[Tuple[Agent, Agent, float]]:
        all_pairings = []
        for agent in self.agents:
            for partner_id, score in agent.partner_preferences.items():
                partner = next(a for a in self.agents if a.id == partner_id)
                mutual_score = score + partner.partner_preferences.get(agent.id, 0)
                all_pairings.append((agent, partner, mutual_score))
        
        sorted_pairings = sorted(all_pairings, key=lambda x: x[2], reverse=True)
        
        unique_pairings = []
        seen = set()
        for agent1, agent2, score in sorted_pairings:
            pair = tuple(sorted([agent1.id, agent2.id]))
            if pair not in seen:
                unique_pairings.append((agent1, agent2, score))
                seen.add(pair)
        
        return unique_pairings[:min(len(unique_pairings), 3)]  # Return top 3 unique pairings

    async def run(self):
        logging.info("Starting Multi-Agent Disco simulation")
        stable_rounds = 0
        previous_pairs = []

        for iteration in range(self.iterations):
            logging.info(f"Starting iteration {iteration + 1}")
            print(f"\n--- Iteration {iteration + 1} ---")
            song = await self.generate_song(random.choice(self.agents))
            print(f"Song: {json.dumps(song, indent=2)}")
            logging.info(f"Generated song: {json.dumps(song)}")

            pairs = self.select_partners()
            print("Dance Pairs:")
            for pair in pairs:
                print(f"  Agent {pair.agent1.id} ({pair.agent1.preferred_style}, {pair.agent1.model}) with Agent {pair.agent2.id} ({pair.agent2.preferred_style}, {pair.agent2.model})")
                logging.info(f"Paired Agent {pair.agent1.id} with Agent {pair.agent2.id}")

            if pairs == previous_pairs:
                stable_rounds += 1
            else:
                stable_rounds = 0
            previous_pairs = pairs

            results = []
            for pair in pairs:
                print(f"\nDance Routine for Agent {pair.agent1.id} and Agent {pair.agent2.id}:")
                routine = await self.collaborate_on_dance(pair, song)
                print(routine)
                score, feedback = await self.judge_dance(routine, random.choice(self.agents))
                result = DanceResult(pair, routine, score, feedback, iteration + 1, song)
                results.append(result)
                self.dance_history.append(result)
                print(f"Score: {score}")
                print(f"Feedback: {feedback}")
                logging.info(f"Dance result - Score: {score}, Feedback: {feedback}")

            self.update_preferences(results)

            if stable_rounds >= self.stabilization_threshold:
                print(f"\nPartner selection stabilized after {iteration + 1} iterations.")
                logging.info(f"Partner selection stabilized after {iteration + 1} iterations.")
                break

        self.generate_final_report()
        self.save_dance_history()

        print("\nHighest-Ranked Pairings:")
        top_pairings = self.calculate_highest_ranked_pairings()
        for i, (agent1, agent2, score) in enumerate(top_pairings, 1):
            print(f"{i}. Agent {agent1.id} ({agent1.preferred_style}, {agent1.model}) and Agent {agent2.id} ({agent2.preferred_style}, {agent2.model})")
            print(f"   Mutual Score: {score}")

    def generate_final_report(self):
        report = "Final Agent Preferences and Best Performances:\n\n"
        
        sorted_agents = sorted(self.agents, key=lambda a: max(a.partner_preferences.values(), default=0), reverse=True)
        
        for agent in sorted_agents:
            report += f"Agent {agent.id} ({agent.preferred_style}, {agent.model}):\n"
            sorted_preferences = sorted(agent.partner_preferences.items(), key=lambda x: x[1], reverse=True)
            for partner_id, preference in sorted_preferences:
                partner = next(a for a in self.agents if a.id == partner_id)
                report += f"  Partner: Agent {partner_id} ({partner.preferred_style}, {partner.model}), Score: {preference}\n"
            
            best_performance = max(
                (result for result in self.dance_history if result.pair.agent1.id == agent.id or result.pair.agent2.id == agent.id),
                key=lambda r: r.score,
                default=None
            )
            
            if best_performance:
                report += f"  Best Performance:\n"
                report += f"    Partner: Agent {best_performance.pair.agent2.id if best_performance.pair.agent1.id == agent.id else best_performance.pair.agent1.id}\n"
                report += f"    Score: {best_performance.score}\n"
                report += f"    Routine:\n{best_performance.routine}\n"
            
            report += "\n"

        report += "Highest-Ranked Pairings:\n\n"
        top_pairings = self.calculate_highest_ranked_pairings()
        for i, (agent1, agent2, score) in enumerate(top_pairings, 1):
            report += f"{i}. Agent {agent1.id} ({agent1.preferred_style}, {agent1.model}) and Agent {agent2.id} ({agent2.preferred_style}, {agent2.model})\n"
            report += f"   Mutual Score: {score}\n\n"

        with open("final_report.txt", "w") as f:
            f.write(report)
        
        logging.info("Final report generated and saved as final_report.txt")

    def save_dance_history(self):
        markdown = "# Dance History\n\n"
        
        for i, result in enumerate(self.dance_history, 1):
            markdown += f"## Dance {i}\n\n"
            markdown += f"### Iteration\n{result.iteration}\n\n"
            markdown += f"### Song\n"
            markdown += f"Title: {result.song.get('title', 'Unknown')}\n"
            markdown += f"Genre: {result.song.get('genre', 'Unknown')}\n"
            markdown += f"Tempo: {result.song.get('tempo', 'Unknown')}\n"
            markdown += f"Mood: {result.song.get('mood', 'Unknown')}\n"
            markdown += f"Description: {result.song.get('description', 'No description available')}\n\n"
            markdown += f"### Pair\n"
            markdown += f"- Agent {result.pair.agent1.id} ({result.pair.agent1.preferred_style}, {result.pair.agent1.model})\n"
            markdown += f"- Agent {result.pair.agent2.id} ({result.pair.agent2.preferred_style}, {result.pair.agent2.model})\n\n"
            markdown += f"### Routine\n{result.routine}\n\n"
            markdown += f"### Score\n{result.score}\n\n"
            markdown += f"### Feedback\n{result.feedback}\n\n"
            markdown += "---\n\n"

        # Adding Highest-Ranked Pairings
        markdown += "# Highest-Ranked Pairings\n\n"
        top_pairings = self.calculate_highest_ranked_pairings()
        for i, (agent1, agent2, score) in enumerate(top_pairings, 1):
            markdown += f"{i}. Agent {agent1.id} ({agent1.preferred_style}, {agent1.model}) and Agent {agent2.id} ({agent2.preferred_style}, {agent2.model})\n"
            markdown += f"   Mutual Score: {score}\n\n"

        # Finding and adding the highest-scored routine
        if self.dance_history:
            highest_scored_routine = max(self.dance_history, key=lambda x: x.score)
            markdown += "# Highest-Ranked Overall Routine\n\n"
            markdown += f"## Score: {highest_scored_routine.score}\n\n"
            markdown += f"### Pair\n"
            markdown += f"- Agent {highest_scored_routine.pair.agent1.id} ({highest_scored_routine.pair.agent1.preferred_style}, {highest_scored_routine.pair.agent1.model})\n"
            markdown += f"- Agent {highest_scored_routine.pair.agent2.id} ({highest_scored_routine.pair.agent2.preferred_style}, {highest_scored_routine.pair.agent2.model})\n\n"
            markdown += f"### Routine\n{highest_scored_routine.routine}\n\n"
            markdown += f"### Feedback\n{highest_scored_routine.feedback}\n\n"

        with open("dance_history.md", "w") as f:
            f.write(markdown)
        
        logging.info("Dance history saved as dance_history.md")

async def main():
    framework = MultiAgentDisco(
        num_agents=6,
        dance_styles=["Salsa", "Tango", "Waltz", "Hip-hop", "Contemporary"],
        models=["vanilj/hermes-3-llama-3.1-8b:latest", "mistral:v0.3", "llama3.1:latest"],  # List of available models
        iterations=10,
        stabilization_threshold=3
    )
    await framework.run()

if __name__ == "__main__":
    asyncio.run(main())
