import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import dotenv from "dotenv";

dotenv.config();

const openAIApiKey = process.env.OPENAI_API_KEY;
const tavilyApiKey = process.env.TAVILY_API_KEY;

async function agent() {
	const agentTools = [new TavilySearch({ maxResults: 3, tavilyApiKey })];
	const agentModel = new ChatOpenAI({ temperature: 0, openAIApiKey });

	const agentCheckpointer = new MemorySaver();
	const agent = createReactAgent({
		llm: agentModel,
		tools: agentTools,
		checkpointSaver: agentCheckpointer,
	});

	const agentFinalState = await agent.invoke(
		{ messages: [new HumanMessage("when and where is the ipl final match 2025")] },
		{ configurable: { thread_id: "42" } }
	);

	console.log(agentFinalState.messages[agentFinalState.messages.length - 1].content);

	const agentNextState = await agent.invoke(
		{ messages: [new HumanMessage("what are the chances of rain for that match")] },
		{ configurable: { thread_id: "42" } }
	);

	console.log(agentNextState.messages[agentNextState.messages.length - 1].content);
}

agent().catch(console.error);
